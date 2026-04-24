package com.example.imagesearch.model;

import lombok.Data;

@Data
public class SearchResult {
    private double score;
    private String productId;
    private String imageUrl;
    private String imageName;
}
