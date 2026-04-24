package com.example.imagesearch.controller;

import com.example.imagesearch.model.SearchResult;
import com.example.imagesearch.service.ImageSearchService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Slf4j
@RestController
@RequestMapping("/api/image")
public class ImageSearchController {

    @Autowired
    private ImageSearchService imageSearchService;

    /**
     * 上传图片到库
     */
    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadImage(
            @RequestParam("file") MultipartFile file) {
        String productId = UUID.randomUUID().toString().replaceAll("-", "");
        String imageName = file.getOriginalFilename();
        // mock 图片存储路径，实际应用中应该将图片保存到文件系统或云存储，并生成对应的 URL
        String imageUrl = "/images/" + productId + "/" + imageName; // 假设图片存储路径为 /images/{productId}/{imageName}
        boolean success = imageSearchService.insertImage(productId, imageUrl, imageName, file);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", success);
        response.put("message", success ? "Image uploaded successfully" : "Upload failed");
        return ResponseEntity.ok(response);
    }

    /**
     * 上传图片到库
     */
    @PostMapping("/uploadByPath")
    public ResponseEntity<Map<String, Object>> uploadByPath(
            @RequestParam("path") String path) throws Exception {
        // 根据path路径批量读取图片文件并上传
        File dir = new File(path);
        if (dir.exists() && dir.isDirectory()) {
            // 读取目录下所有图片文件
            File[] files = dir.listFiles();
            for (File file : files) {
                String imageName = file.getName();
                if (file.isFile() && (imageName.endsWith(".jpg") || imageName.endsWith(".png"))) {
                    String productId = UUID.randomUUID().toString().replaceAll("-", "");
                    String imageUrl = "/images/" + productId + "/" + imageName; // 假设图片存储路径为 /images/{productId}/{imageName}
                    // file 转 MultipartFile
                    String contentType = "image/jpeg"; // 根据实际情况设置内容类型
                    if (imageName.endsWith(".png")) {
                        contentType = "image/png";
                    }
                    MultipartFile multipartFile = new MockMultipartFile(imageName, imageName, contentType, new FileInputStream(file));
                    boolean success = imageSearchService.insertImage(productId, imageUrl, imageName, multipartFile);
                    log.info("Upload {}: {}", imageName, success ? "success" : "failed");
                }
            }
        }
        
        Map<String, Object> response = new HashMap<>();
        return ResponseEntity.ok(response);
    }

    /**
     * 以图搜图
     */
    @PostMapping("/search")
    public ResponseEntity<Map<String, Object>> searchSimilar(@RequestParam("file") MultipartFile file) {
        List<SearchResult> results = imageSearchService.searchSimilar(file);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("count", results.size());
        response.put("results", results);
        
        return ResponseEntity.ok(response);
    }

    /**
     * 获取统计信息
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        long count = imageSearchService.getCollectionCount();
        
        Map<String, Object> response = new HashMap<>();
        response.put("collectionCount", count);
        response.put("status", "running");
        
        return ResponseEntity.ok(response);
    }
}