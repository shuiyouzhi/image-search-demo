package com.example.imagesearch.service;

import ai.onnxruntime.*;
import lombok.extern.slf4j.Slf4j;
import org.imgscalr.Scalr;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

@Slf4j
@Service
public class FeatureExtractor {

    @Value("${model.path}")
    private String modelPath;

    @Value("${model.input-size}")
    private int inputSize;

    @Value("${model.input-name}")
    private String inputName;

    @Value("${model.output-name}")
    private String outputName;

    private OrtEnvironment environment;
    private OrtSession session;

    @PostConstruct
    public void init() throws OrtException, IOException {
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 从resources目录加载模型
        var modelStream = getClass().getClassLoader().getResourceAsStream(modelPath);
        if (modelStream == null) {
            throw new IOException("Model not found: " + modelPath);
        }

        Path tempModel = Files.createTempFile("model", ".onnx");
        Files.copy(modelStream, tempModel, java.nio.file.StandardCopyOption.REPLACE_EXISTING);

        session = environment.createSession(tempModel.toString(), sessionOptions);
        log.info("Model loaded successfully");

        Files.delete(tempModel);
    }

    @PreDestroy
    public void destroy() throws OrtException {
        if (session != null)
            session.close();
        if (environment != null)
            environment.close();
    }

    public float[] extractFeature(MultipartFile imageFile) throws IOException, OrtException {
        return extractFeature(imageFile.getBytes());
    }

    public float[] extractFeature(byte[] imageBytes) throws IOException, OrtException {
        // 1. 预处理图片（使用纯 Java 实现）
        float[] inputData = preprocessImage(imageBytes);

        // 2. 创建输入张量
        long[] inputShape = { 1, 3, inputSize, inputSize };
        OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(inputData), inputShape);
        
        // 3. 推理
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

        OnnxValue outputValue = result.get(outputName).get();

        // 4. 获取特征向量
        float[][] output = (float[][]) outputValue.getValue();
        float[] features = output[0];

        // 5. L2 归一化
        normalize(features);

        result.close();
        inputTensor.close();

        return features;
    }

    private float[] preprocessImage(byte[] imageBytes) throws IOException {
        // 读取图片
        BufferedImage originalImage = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (originalImage == null) {
            throw new IOException("Failed to read image");
        }

        // 确保图片是 RGB 格式
        BufferedImage rgbImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(),
                BufferedImage.TYPE_INT_RGB);
        rgbImage.getGraphics().drawImage(originalImage, 0, 0, null);

        // 缩放图片到目标尺寸
        BufferedImage resizedImage = Scalr.resize(rgbImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, inputSize,
                inputSize);

        // 获取实际尺寸（确保与 inputSize 一致）
        int width = resizedImage.getWidth();
        int height = resizedImage.getHeight();

        // 转换为 RGB 数组并归一化
        float[] inputData = new float[3 * inputSize * inputSize];

        for (int y = 0; y < height && y < inputSize; y++) {
            for (int x = 0; x < width && x < inputSize; x++) {
                int rgb = resizedImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // 归一化并使用 ImageNet 均值/标准差
                inputData[0 * inputSize * inputSize + y * inputSize + x] = ((r / 255.0f) - 0.485f) / 0.229f;
                inputData[1 * inputSize * inputSize + y * inputSize + x] = ((g / 255.0f) - 0.456f) / 0.224f;
                inputData[2 * inputSize * inputSize + y * inputSize + x] = ((b / 255.0f) - 0.406f) / 0.225f;
            }
        }

        return inputData;
    }

    private void normalize(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        float norm = (float) Math.sqrt(sum);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}