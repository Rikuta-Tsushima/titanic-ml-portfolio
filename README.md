# Titanic Survival Prediction — Portfolio

教科書どおりの前処理＋3モデル（LogReg / SVC / RandomForest）で比較。
**目的**: リーク対策を守った再現性あるワークフローを作り、面接で説明できる形にする。

## 前処理（trainのみでfit → 両データにtransform）
- Embarked: "C" で補完（方針固定）
- Age: `Pclass×Sex` の中間値で補完
- Fare: `Pclass×Embarked` の中間値で補完
- 変換: `Fare_log1p`

## 特徴量
- FamilySize, IsAlone, WomanChild, MotherLite  
- TicketGroupSize, TicketIsShared, FarePerPerson_log1p  
- Pclass_Sex（相互作用カテゴリ）

## モデル比較（Stratified 5-fold CV）
- Logistic Regression: ACC ≈ **0.8305**, AUC ≈ 0.8657
- SVC (RBF, tuned):  ACC ≈ 0.8294,  AUC ≈ 0.8658
- RandomForest (tuned): **ACC ≈ 0.8462**, **AUC ≈ 0.8741**
- Kaggle Public LB: 0.76076

## 再現方法
1) `pip install -r requirements.txt`  
2) `data/train.csv`, `data/test.csv` を配置（リポには含めません）  
3) `notebooks/titanic_eda.ipynb` を実行（順次追加予定）

## 学び / 工夫 / 反省
- **fit と transform を分離**してリーク防止（ColumnTransformer + Pipeline）
- `merge` によるグループ中間値補完、`FarePerPerson_log1p` でグループ購入の歪み補正
- しきい値調整や汎化重視のRF設定は今後の改善ポイント
