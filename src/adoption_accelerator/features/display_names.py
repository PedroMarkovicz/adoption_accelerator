"""
Mapping for converting internal technical feature names to user-friendly display names.
Used by the interpretation layer to ensure explanations are understandable by shelter operators.
"""

USER_DISPLAY_NAMES = {
    # Tabular - Core
    "Age": "Pet Age (Months)",
    "Fee": "Adoption Fee",
    "Gender": "Gender",
    "Quantity": "Number of Pets in Listing",
    "MaturitySize": "Expected Adult Size",
    "FurLength": "Fur Length",
    "PhotoAmt": "Number of Photos",
    "VideoAmt": "Number of Videos",
    "has_photos": "Has Photos",
    "Breed1": "Primary Breed",
    "Breed2": "Secondary Breed",
    "Color1": "Primary Color",
    "Color2": "Secondary Color",
    "Color3": "Tertiary Color",
    "Type": "Pet Type",
    "State": "State",

    # Tabular - Health
    "Vaccinated": "Vaccination Status",
    "Dewormed": "Deworming Status",
    "Sterilized": "Sterilization Status",
    "Vaccinated_tristate": "Vaccination Status",
    "Dewormed_tristate": "Deworming Status",
    "Sterilized_tristate": "Sterilization Status",
    "health_care_score": "Health Care Score",

    # Tabular - Derived
    "log1p_PhotoAmt": "Number of Photos",
    "log_photo_amt": "Number of Photos",
    "log1p_Age": "Pet Age",
    "log_age": "Pet Age",
    "log_fee": "Adoption Fee",
    "is_dog": "Pet Type (Dog vs Cat)",
    "is_free_adoption": "Free Adoption",
    "fee_per_pet": "Adoption Fee per Pet",
    "age_bin": "Age Group",
    "age_x_type": "Age-Species Interaction",
    "age_x_fee": "Age-Fee Interaction",
    "photo_x_desc": "Photos-Description Interaction",

    # Tabular - Rescuer Statistics
    "rescuer_pet_count": "Rescuer's Listing History",
    "rescuer_mean_fee": "Rescuer's Average Fee",
    "rescuer_mean_photo_amt": "Rescuer's Average Photos per Listing",
    "rescuer_mean_age": "Rescuer's Average Pet Age",
    "rescuer_std_fee": "Rescuer's Fee Consistency",

    # Tabular - Frequency Encodings
    "breed1_freq_encoded": "Breed Popularity",
    "breed1_frequency": "Breed Popularity",
    "breed2_frequency": "Secondary Breed Popularity",
    "color1_frequency": "Color Popularity",
    "state_frequency": "State Listing Volume",

    # Image
    "mean_crop_confidence": "Photo Composition Quality",
    "mean_blur_score": "Photo Clarity",
    "mean_image_brightness": "Photo Brightness",

    # Text - Basic
    "description_length": "Description Length",
    "word_count": "Description Word Count",
    "n_sentences": "Description Sentence Count",
    "sentence_count": "Sentence Count",
    "mean_word_length": "Average Word Length",
    "avg_word_length": "Average Word Length",
    "uppercase_ratio": "Uppercase Letter Frequency",

    # Text - Sentiment
    "doc_sentiment_score": "Description Sentiment",
    "doc_sentiment_magnitude": "Description Emotional Intensity",
    "sentiment_variance": "Description Sentiment Consistency",
    "sentence_count_sentiment": "Description Sentence Detail",

    # Text - NLP / Metadata
    "entity_count": "Description Detail Level",
    "entity_type_count": "Description Entity Variety",

    # Aggregated Embeddings
    "text_semantic_patterns": "Description Content",
    "image_visual_patterns": "Photo Content",
}
