# Titanic Survival Prediction — Portfolio

教科書どおりの前処理と3モデル（LogReg / SVC / RandomForest）を同一条件で比較。
- **前処理**: trainのみで統計をfit → train/testへ同じtransform  
  - Age: Pclass×Sexの中央値 / Fare: Pclass×Embarkedの中央値  
  - 変換: `Fare_log1p`
- **特徴(Tier1, Titleなし)**: FamilySize, IsAlone, WomanChild, MotherLite,
  TicketGroupSize, TicketIsShared, FarePerPerson_log1p, Pclass_Sex
- **CV(5-fold)**: LogReg ACC≈0.8305 / SVC ACC≈0.8294 / RF ACC≈0.8462, AUCはRFが最良
- **Kaggle例**: 0.76076

再現: `pip install -r requirements.txt`、`data/train.csv` と `data/test.csv` を配置、`notebooks/` を上から実行。
