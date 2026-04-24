package com.example.imagesearch.service;

import com.example.imagesearch.model.SearchResult;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.grpc.GetCollectionStatisticsResponse;
import io.milvus.grpc.MutationResult;
import io.milvus.grpc.SearchResults;
import io.milvus.param.*;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import java.util.*;

@Slf4j
@Service
public class ImageSearchService {

    @Autowired
    private MilvusServiceClient milvusClient;

    @Autowired
    private FeatureExtractor featureExtractor;

    @Value("${milvus.database-name:image_search_db}")
    private String databaseName;

    @Value("${milvus.collection-name:product_images}")
    private String collectionName;

    @Value("${milvus.vector-dim:1280}")
    private int vectorDim;

    @Value("${search.top-k:5}")
    private int topK;

    @Value("${search.nprobe:10}")
    private int nprobe;

    /**
     * 获取集合中的记录数量
     */
    public long getCollectionCount() {
        try {
            R<GetCollectionStatisticsResponse> resp = milvusClient.getCollectionStatistics(
                    GetCollectionStatisticsParam.newBuilder()
                            .withDatabaseName(databaseName)
                            .withCollectionName(collectionName)
                            .build());
            if (resp.getStatus() == R.Status.Success.getCode()) {
                // 从 stats 列表中获取行数
                for (io.milvus.grpc.KeyValuePair kv : resp.getData().getStatsList()) {
                    if ("row_count".equals(kv.getKey())) {
                        return Long.parseLong(kv.getValue());
                    }
                }
            }
            log.error("Failed to get collection count: {}", resp.getMessage());
            return 0;
        } catch (Exception e) {
            log.error("Error getting collection count", e);
            return 0;
        }
    }

    @PostConstruct
    public void init() {
        createCollectionIfNotExists();
    }

    private void createCollectionIfNotExists() {
        R<Boolean> hasResp = milvusClient.hasCollection(
                HasCollectionParam.newBuilder()
                        .withDatabaseName(databaseName)
                        .withCollectionName(collectionName)
                        .build());

        if (hasResp.getData()) {
            log.info("Collection {} already exists", collectionName);
            return;
        }

        // 构建 Schema - Milvus 2.5.x 使用 addFieldType 方法
        CreateCollectionParam createParam = CreateCollectionParam.newBuilder()
                .withDatabaseName(databaseName)
                .withCollectionName(collectionName)
                .withDescription("Product image search collection")
                .withShardsNum(2)
                .addFieldType(FieldType.newBuilder()
                        .withName("id")
                        .withDataType(DataType.Int64)
                        .withPrimaryKey(true)
                        .withAutoID(true)
                        .build())
                .addFieldType(FieldType.newBuilder()
                        .withName("product_id")
                        .withDataType(DataType.VarChar)
                        .withMaxLength(100)
                        .build())
                .addFieldType(FieldType.newBuilder()
                        .withName("image_url")
                        .withDataType(DataType.VarChar)
                        .withMaxLength(500)
                        .build())
                .addFieldType(FieldType.newBuilder()
                        .withName("image_name")
                        .withDataType(DataType.VarChar)
                        .withMaxLength(200)
                        .build())
                .addFieldType(FieldType.newBuilder()
                        .withName("embedding")
                        .withDataType(DataType.FloatVector)
                        .withDimension(vectorDim)
                        .build())
                .build();

        R<RpcStatus> createResp = milvusClient.createCollection(createParam);

        if (createResp.getStatus() == R.Status.Success.getCode()) {
            log.info("Collection {} created successfully", collectionName);
            createIndex();
            loadCollection();
        } else {
            log.error("Failed to create collection: {}", createResp.getMessage());
        }
    }

    private void createIndex() {
        CreateIndexParam indexParam = CreateIndexParam.newBuilder()
                .withDatabaseName(databaseName)
                .withCollectionName(collectionName)
                .withFieldName("embedding")
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.COSINE)
                .withExtraParam("{\"nlist\": 1024}") // IVF_FLAT 索引需要 nlist 参数
                .build();

        R<RpcStatus> indexResp = milvusClient.createIndex(indexParam);

        if (indexResp.getStatus() == R.Status.Success.getCode()) {
            log.info("Index created successfully");
        } else {
            log.error("Failed to create index: {}", indexResp.getMessage());
        }
    }

    private void loadCollection() {
        LoadCollectionParam loadParam = LoadCollectionParam.newBuilder()
                .withDatabaseName(databaseName)
                .withCollectionName(collectionName)
                .build();

        R<RpcStatus> loadResp = milvusClient.loadCollection(loadParam);

        if (loadResp.getStatus() == R.Status.Success.getCode()) {
            log.info("Collection loaded successfully");
        } else {
            log.error("Failed to load collection: {}", loadResp.getMessage());
        }
    }
    // 特征提取和数据插入
    public boolean insertImage(String productId, String imageUrl, String imageName, MultipartFile imageFile) {
        try {
            float[] embedding = featureExtractor.extractFeature(imageFile);

            // 将 float[] 转换为 List<Float>
            List<Float> embeddingList = new ArrayList<>(embedding.length);
            for (float v : embedding) {
                embeddingList.add(v);
            }

            List<InsertParam.Field> fields = new ArrayList<>();
            fields.add(new InsertParam.Field("product_id", Collections.singletonList(productId)));
            fields.add(new InsertParam.Field("image_url", Collections.singletonList(imageUrl)));
            fields.add(new InsertParam.Field("image_name", Collections.singletonList(imageName)));
            fields.add(new InsertParam.Field("embedding", Collections.singletonList(embeddingList)));

            InsertParam insertParam = InsertParam.newBuilder()
                    .withDatabaseName(databaseName)
                    .withCollectionName(collectionName)
                    .withFields(fields)
                    .build();

            R<MutationResult> insertResp = milvusClient.insert(insertParam);

            if (insertResp.getStatus() == R.Status.Success.getCode()) {
                log.info("Image inserted successfully: {}", imageName);
                return true;
            } else {
                log.error("Failed to insert image: {}", insertResp.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("Error inserting image", e);
            return false;
        }
    }

    // Top k 近邻搜索, 返回相似图片列表
    public List<SearchResult> searchSimilar(MultipartFile queryImage) {
        try {
            // 题
            float[] queryVector = featureExtractor.extractFeature(queryImage);

            // 将 float[] 转换为 List<Float>
            List<Float> queryVectorList = new ArrayList<>(queryVector.length);
            for (float v : queryVector) {
                queryVectorList.add(v);
            }

            SearchParam searchParam = SearchParam.newBuilder()
                    // 指定数据库和集合
                    .withDatabaseName(databaseName)
                    .withCollectionName(collectionName)

                    // 指定集合中进行向量搜索的字段名称
                    .withVectorFieldName("embedding")

                    // 待搜索的向量列表（支持批量搜索），每个向量是一个浮点数列表
                    .withVectors(Collections.singletonList(queryVectorList))

                    // 返回与查询向量最相似的 K 条结果
                    .withTopK(topK)

                    // 向量距离计算方法，影响搜索结果的相似度判断. COSINE、L2、IP 等
                    // COSINE：余弦相似度（范围 -1~1，越大越相似）- 适合文本、图像相似度
                    // L2：欧氏距离（非负，越小越相似）- 适合空间坐标、聚类分析
                    // IP：内积（无界，越大越相似）- 适合推荐系统、词嵌入
                    .withMetricType(MetricType.COSINE)

                    // 搜索时要探测的聚类单元数量,值越大，精度越高但速度越慢。对于 IVF 索引类型必须设置 nprobe 参数，默认值为 1。
                    .withParams("{\"nprobe\": " + nprobe + "}")

                    // 指定除向量外的标量字段，需要返回的具体值
                    .withOutFields(Arrays.asList("product_id", "image_url", "image_name"))
                    .build();

            R<SearchResults> searchResp = milvusClient.search(searchParam);

            if (searchResp.getStatus() != R.Status.Success.getCode()) {
                log.error("Search failed: {}", searchResp.getMessage());
                return Collections.emptyList();
            }

            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResp.getData().getResults());
            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(0);

            List<SearchResult> results = new ArrayList<>();
            for (SearchResultsWrapper.IDScore score : scores) {
                SearchResult result = new SearchResult();
                result.setScore(score.getScore());
                result.setProductId((String) score.get("product_id"));
                result.setImageUrl((String) score.get("image_url"));
                result.setImageName((String) score.get("image_name"));
                results.add(result);
            }

            log.info("Found {} similar images", results.size());
            return results;

        } catch (Exception e) {
            log.error("Error searching similar images", e);
            return Collections.emptyList();
        }
    }
}