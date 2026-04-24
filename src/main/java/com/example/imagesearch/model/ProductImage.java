package com.example.imagesearch.model;

import lombok.Data;

@Data
public class ProductImage {
    private Long id;
    private String productId;
    private String imageUrl;
    private String imageName;
}