# Titanic Survival Prediction — Portfolio

教科書どおりの前処理＋3モデル（LogReg / SVC / RandomForest）で比較。
**目的**: リーク対策を守った再現性あるワークフローを作り、面接で説明できる形にする。

## EDAで気づいたこと（傾向と解釈）

- **性別（Sex）**：女性の生存率が男性より圧倒的に高い  
  └ 「女性と子ども優先」の救助方針の影響が強い

- **等級（Pclass）**：1等 > 2等 > 3等 の順に生存率が高い  
  └ 甲板/客室の位置や救命ボートまでのアクセス、乗客層の違い

- **年齢（Age）**：子ども（おおむね16歳未満）は生存率が高い  
  └ 方針 + 付き添い（保護者）による誘導効果

- **家族構成（FamilySize / IsAlone）**：  
  中規模（2〜4人）は有利、**単独（IsAlone=1）**と**大家族（>4）**は不利になりがち  
  └ 情報共有・助け合い vs. 移動の難しさ

- **乗船港（Embarked）**：**C（Cherbourg）**発の生存率が相対的に高く、**S（Southampton）**が低い傾向  
  └ Pclass・運賃分布の違いが背景にある（Cは上位クラスが多い）

- **運賃（Fare）**：高いほど生存傾向。**上位1%はほぼPclass=1**  
  └ 客室位置・クラスに強く相関。分布は右に長い（→ `log1p` で扱いやすく）

- **同一チケット（TicketGroupSize）**：>1 は**グループ行動**の影響が出る  
  └ 家族や同行者の有無が救助にプラス/マイナスの両面で効く

- **欠損**：Age に欠損が目立つ / Cabin は欠損が多く特徴化しづらい  
  └ Age は **`Pclass×Sex` の中央値**で補完、Cabinは今回未使用

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

## 学び
- **fit と transform を分離**してリーク防止（ColumnTransformer + Pipeline）
- グループ補完の実装パターン：groupby().median() を merge で付与→maskで欠損行のみ置換。
## 工夫  
- **グループ補完**：`Age` は `Pclass×Sex`、`Fare` は `Pclass×Embarked` の**中間値**で `merge` 補完  
- **右裾の圧縮**：`Fare_log1p` を追加  
- `merge` によるグループ中間値補完、`FarePerPerson_log1p` でグループ購入の歪み補正
## 反省
- しきい値調整や汎化重視のRF設定は今後の改善ポイント
- 前処理の完全Pipeline化：グループ中間値や分位点計算、フラグ生成まで カスタムTransformer にして fold内fit
