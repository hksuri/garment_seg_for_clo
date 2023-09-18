
deepfashion_dict_grams= {}

deepfashion_dict_grams["short sleeve top"]= 160
deepfashion_dict_grams["long sleeve top"] = 230
deepfashion_dict_grams["short sleeve outwear"] = 490 
deepfashion_dict_grams["long sleeve outwear"] = 590
deepfashion_dict_grams["vest"] = 150
deepfashion_dict_grams["sling"] = 300
deepfashion_dict_grams["shorts"] = 260
deepfashion_dict_grams["trousers"] = 550 
deepfashion_dict_grams["skirt"] = 250
deepfashion_dict_grams["short sleeve dress"] = 350
deepfashion_dict_grams["long sleeve dress"] = 450
deepfashion_dict_grams["vest dress"] = 320
deepfashion_dict_grams["sling dress"]= 400


def get_cloth(cloth_items=None, cloth_areas=None, A_cov0=None):
    grams= 0.
    for cloth in cloth_items:
        grams += deepfashion_dict_grams[cloth]  
    A_cov1 = 0
    for cloth in cloth_areas:
        A_cov1 += cloth

    # A_cov1 = *A_cov1
    clo =0.919 + (0.255*0.001*grams) - (0.00874*A_cov0) - (0.00510*A_cov1)
    return (clo)


