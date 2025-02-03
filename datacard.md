# Jigsaw Toxic Comment Classification Dataset

## Overview
Version: 1.0
Date Created: 2025-02-03

### Description

        The Jigsaw Toxic Comment Classification Dataset is designed to help identify and classify toxic online comments.
        It contains text comments with multiple toxicity-related labels including general toxicity, severe toxicity,
        obscenity, threats, insults, and identity-based hate speech.

        The dataset includes:
        1. Main training data with binary toxicity labels
        2. Unintended bias training data with additional identity attributes
        3. Processed versions with sequence length 128 for direct model input
        4. Test and validation sets for model evaluation

        This dataset was created by Jigsaw and Google's Conversation AI team to help improve online conversation quality
        by identifying and classifying various forms of toxic comments.
        

## Column Descriptions

- **id**: Unique identifier for each comment
- **comment_text**: The text content of the comment to be classified
- **toxic**: Binary label indicating if the comment is toxic
- **severe_toxic**: Binary label for extremely toxic comments
- **obscene**: Binary label for obscene content
- **threat**: Binary label for threatening content
- **insult**: Binary label for insulting content
- **identity_hate**: Binary label for identity-based hate speech
- **target**: Overall toxicity score (in bias dataset)
- **identity_attack**: Binary label for identity-based attacks
- **identity_***: Various identity-related attributes in the bias dataset
- **lang**: Language of the comment

## Files

