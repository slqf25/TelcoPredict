# Telco Customer Churn Prediction

## 1. System Knowledge｜系统知识

**中文**

我们的系统根据电讯客户目前的合约、tenure（已经使用服务的月数）、收费、付款方式和订阅服务，估计客户的 churn risk。Churn 指客户停止使用或离开公司服务。这个 prediction 是给业务人员参考的 risk signal，不是保证客户一定会离开，也不会自动决定谁应该得到优惠。

**English**

Our system uses a telco customer's contract, tenure, charges, payment method and subscribed services to estimate churn risk. Churn means that the customer stops using or leaves the service. The prediction is a risk signal for business support; it does not guarantee that the customer will leave or automatically decide who receives an offer.

---

**Dataset and system flow｜数据与系统流程**

The dataset contains **7,043 customers** and **21 original attributes**. Approximately **26.54% churned** and **73.46% were retained**.

```text
Historical customer data
→ cleaning and feature engineering
→ stratified 80/20 train-test split
→ scaling and SMOTE on training data
→ train four models
→ evaluate on the same 1,409 test customers
→ place fitted models in Streamlit
→ enter a customer profile
→ produce a churn-risk result
```

**中文**

Training data 用来让模型学习；test data 是模型训练完成后才使用的“考试资料”。两者必须分开，否则模型可能只是记住旧客户，而不是真的会处理新客户。

**English**

Training data is used for learning. Test data is the “exam data” used only after training. They must remain separate; otherwise, the model may simply memorise old customers instead of handling new customers.

---

**Stratified 80/20 split｜分层训练测试切分**

**中文**

80% 数据用于训练，20% 用于最终测试。Stratified 表示两边都尽量保留接近 26.54% 的 churn proportion，避免其中一边刚好有太多或太少 churners。80/20 是训练数据量和测试可信度之间的实际平衡，不是唯一正确比例。

**English**

Eighty percent of the data is used for training and twenty percent for final testing. Stratified means that both sides keep approximately the same 26.54% churn proportion, avoiding a group with too many or too few churners. The 80/20 split is a practical balance, not the only valid ratio.

---

**Five-fold cross-validation｜五折交叉验证**

**中文**

一次切分可能碰巧容易或困难，所以我们在 training partition 内再分成五份。每一轮用四份训练、另一份验证；五轮后，每份都当过一次 validation fold。Final test set 不参与这五轮。

```text
Round 1: validate Fold 1; train on Folds 2, 3, 4, 5
Round 2: validate Fold 2; train on Folds 1, 3, 4, 5
Round 3: validate Fold 3; train on Folds 1, 2, 4, 5
Round 4: validate Fold 4; train on Folds 1, 2, 3, 5
Round 5: validate Fold 5; train on Folds 1, 2, 3, 4
```

为什么是 5：

- `k=5`：每轮约 80% fold data 用于训练，同时只需重复五次，稳定性和计算时间较平衡。
- `k=2`：可以运行，但每轮只有一半 fold data 训练，而且只有两个 validation results。
- `k=10`：也可以，但每个模型要训练十轮，GridSearch 的重复训练量约增加一倍；每个 validation fold 也更小。
- `k=0`：没有 fold，无法运行。
- `k=1`：没有另一份独立 validation fold，因此不构成 cross-validation。

**English**

One split may accidentally be easy or difficult, so we divide the training partition into five folds. In each round, four folds are used for training and one for validation. Every fold is used for validation once, while the final test set remains untouched.

We choose five because it balances validation reliability and computational cost. Two folds provide very few validation results and use only half of the fold data for training. Ten folds are valid but require roughly twice as many repeated training runs and create smaller validation folds. Zero or one fold cannot form cross-validation.

**If asked｜老师问时**

> We use five folds because it provides a practical balance between validation reliability and computational cost. Two folds provide too few validation results, while ten folds require roughly twice as many repeated training runs. Zero or one fold cannot form cross-validation.

---

**SMOTE｜处理类别不平衡**

**中文**

Churn customers 只有约 26.54%，模型可能因此习惯预测 No Churn。SMOTE 参考现有 churn customers 的 feature patterns，在 training data 中建立 synthetic minority examples。它不是创造真实客户，也不能在 split 前执行，否则 synthetic information 可能进入 test data，形成 data leakage（像模型间接看到考试内容）。

**English**

Only about 26.54% of customers churned, so the model may become too comfortable predicting No Churn. SMOTE uses existing churn feature patterns to create synthetic minority examples in training data. It does not create real customers. It must not be applied before the split because synthetic information could enter the test data and cause data leakage.

---

**Current model features｜当前模型实际使用的 Features**

Feature 是模型实际接收的 input。Streamlit 的客户表单经过 encoding（把文字类别转成 numeric columns）后，形成以下 23 个 predictors。

| Group | Current features | 中文解释｜Simple meaning |
|---|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | 性别、长者、partner、dependents 状态 |
| Account | `tenure` | 已使用服务多少个月｜Months with the company |
| Phone | `PhoneService`, `MultipleLines` | 是否有电话与多线服务 |
| Internet add-ons | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | 是否订阅六种网络附加服务 |
| Billing | `PaperlessBilling`, `MonthlyCharges`, `TotalCharges` | 电子账单、月费、累积收费 |
| Engineered | `ContractRiskScore`, `ChargesToTenureRatio` | 根据原始资料计算的新 features |
| Internet type | `InternetService_Fiber optic`, `InternetService_No` | DSL 是 reference；两个 columns 都为 0 就代表 DSL |
| Payment | `PaymentMethod_Credit card (automatic)`, `PaymentMethod_Electronic check`, `PaymentMethod_Mailed check` | Bank transfer automatic 是 reference |

`ContractRiskScore` uses Month-to-month = 2, One year = 1 and Two year = 0. `ChargesToTenureRatio` is `MonthlyCharges / (tenure + 1)`.

**中文**

Reference category 的意思是不用为每个类别都建立 column。例如 Internet 有 DSL、Fiber 和 No 三类；Fiber=0 且 No=0 时已经代表 DSL，所以不需要第三个重复 column。

**English**

A reference category means that we do not need a separate column for every category. Internet has DSL, Fiber and No. When Fiber and No are both zero, the customer is DSL, so a third repeated column is unnecessary.

---

**Which current features could be removed?｜当前还有哪些 Feature 可以考虑删除？**

**中文**

Candidate 不等于立即删除。正确做法是 feature ablation：删除一个 feature、重新训练，再使用相同 split 和 parameters 比较 Recall、F1、AUC 与 fold stability。

1. **First candidate — `gender`**：Cramér’s V = 0.008、p-value = 0.487，几乎没有单变量 churn association，也可以减少不必要的 demographic/fairness concern。
2. **Second candidate — `PhoneService`**：V = 0.011、p = 0.339，也非常弱；但它区分“没有电话”和“有电话但没有 multiple lines”，删除前必须检查 encoding。
3. **Later candidates**：`MultipleLines` (V=0.040)、`StreamingMovies` (0.061)、`StreamingTV` (0.063)、`DeviceProtection` (0.066)、`OnlineBackup` (0.082)。它们单独关系很弱，但可能通过 interaction 帮助 tree models。

**English**

A candidate is not an immediate removal. We should perform feature ablation: remove one feature, retrain the model with the same split and parameters, and compare Recall, F1, AUC and fold stability.

`gender` is the first candidate because it has Cramér’s V of 0.008 and p-value of 0.487, showing almost no individual churn association. `PhoneService` is a second candidate, but removing it may make no-phone customers difficult to distinguish from phone customers without multiple lines. Several service features are later candidates, but they may still help tree models through interactions.

**If asked for one feature｜如果老师只要一个答案**

> Among the current 23 predictors, gender is our first removal candidate because it shows almost no univariate churn association. However, we should confirm the decision through an ablation test before changing the final model.

---

**VIF and multicollinearity｜特征信息重叠**

**中文**

VIF（Variance Inflation Factor）不是检查 feature 与 churn 的关系。它检查一个 input feature 能不能被其他 input features 大量推测出来。

例如同时使用每天、每周和每月学习小时预测成绩，这三个 columns 可以互相换算，资料高度重复。这就是 multicollinearity。

- VIF 接近 1：几乎没有重叠。
- VIF 1–5：通常较低。
- VIF 5–10：需要注意。
- VIF > 10：严重重叠。

Streamlit 当前画面显示 `MonthlyCharges` VIF 约 866.1。它不代表与 churn 有 866 倍关系，也不是 Accuracy 下降 866%。根据 `VIF = 1/(1-R²)`，866.1 对应约 99.88% 的 auxiliary explained variation。不过当前函数直接对 encoded matrix 计算而没有加入 intercept，因此应重视“重叠非常严重”这个结论，不应过度解读 exact value。

它很高是因为 Internet type、phone、add-ons、TotalCharges 和 ChargesToTenureRatio 都包含月费相关信息。Notebook 在 scaled、SMOTE-resampled training representation 上得到约 20.17；计算 context 不同，所以 exact value 不同，但两者都大于 10，都说明严重重叠。

**English**

VIF does not measure the relationship between a feature and churn. It checks whether one input feature can be strongly predicted from the other input features.

A VIF near 1 means little overlap; values above 5 require attention; values above 10 indicate severe overlap. The current Streamlit view shows a MonthlyCharges VIF of about 866.1. This does not mean an 866-times relationship with churn or an 866% Accuracy drop. The calculation uses the encoded matrix without an intercept, so the robust conclusion is severe overlap rather than a direct business interpretation of the exact number.

MonthlyCharges overlaps with internet type, phone service, add-ons, TotalCharges and ChargesToTenureRatio. The notebook uses a different scaled and SMOTE-resampled calculation context and shows about 20.17. Both values exceed 10 and support the same conclusion of severe overlap.

High VIF mainly limits independent Logistic Regression coefficient interpretation. Tree-model predictions are less sensitive, but correlated features may share importance. We should not immediately remove MonthlyCharges because it remains a top-four Random Forest feature. `TotalCharges` can also be tested for removal, but it is a top-five Random Forest feature, so neither should be removed without ablation evidence.

**If asked｜老师问时**

> VIF measures overlap among predictors, not their relationship with churn. MonthlyCharges has a very high VIF because service selections and other charge-related variables contain similar information. High VIF does not automatically mean that MonthlyCharges should be removed; we should compare ablation models first.

## 2. Model Explanation｜模型解释

**Logistic Regression**

**中文**

Logistic Regression 是容易解释的 linear baseline。它为每个 feature 学习一个 coefficient（weight），把所有 weights 组合成 score，再转换成 0–1 risk output。正 coefficient 通常把 modelled risk 推高，负 coefficient 推低，但不证明 causation。

Parameter：

- `max_iter=2000`：电脑最多重复调整 coefficients 2,000 次，让 optimisation 有足够机会稳定。它不是 2,000 层，也不是 2,000 features。
- Numerical scaling：把 tenure、charges 等不同单位放到较可比较的 scale，帮助 optimisation。

为什么选它：作为 interpretable baseline，检查复杂模型是否真的带来改善。结果为 Accuracy 75.23%、Recall 71.93%、F1 60.65%、AUC 0.833。优点是容易说明方向；限制是可能错过 nonlinear patterns，而且高 VIF 会令 coefficient 不稳定。

**English**

Logistic Regression is our interpretable linear baseline. It learns a coefficient, or weight, for every feature, combines the weights into a score and converts the score into a 0–1 risk output. A positive coefficient generally pushes modelled risk upward, while a negative one pushes it downward, but it does not prove causation.

`max_iter=2000` allows the computer up to 2,000 optimisation iterations to find a stable solution. It does not mean 2,000 layers or features. Numerical scaling places variables such as tenure and charges on more comparable scales.

We include it as an interpretable baseline. It achieved 71.93% Recall, F1 of 60.65% and AUC of 0.833. Its strength is interpretation; its limitation is that it may miss nonlinear patterns and its coefficients are affected by high VIF.

---

**Decision Tree**

**中文**

Decision Tree 像连续问问题，例如“tenure 是否小于 12？”每个问题是 decision rule；提出问题的位置是 node；根据答案分开叫 split；路线终点叫 leaf，模型在 leaf 给 prediction。

Parameters：

- `max_depth=5`：每条从树顶到 leaf 的路线最多约五层，不是整棵树只能有五个问题。太小会 underfit；太大会建立过细规则而 overfit。5 让模型保留 nonlinear ability，同时限制复杂度并保持规则可读。
- `min_samples_split=10`：一个 node 至少有 10 个 training samples 才能继续分，避免只根据少数客户建立新问题。
- `random_state=42`：固定随机过程，让重复运行得到可重现结果；42 本身没有特殊数学优势。

为什么选它：提供容易解释的 nonlinear model，并与 ensemble trees 比较。它的 Recall 最高，为 77.01%，代表抓到最多真实 churners；但 Precision 只有 50.35%，会产生更多 false alerts，而且单棵树较不稳定。

**English**

A Decision Tree asks a sequence of questions, such as whether tenure is below 12. Each question is a decision rule, the question point is a node, dividing customers is a split, and the final endpoint is a leaf where the prediction is made.

`max_depth=5` allows about five levels along each path. A very small depth may underfit, while a very large depth may create overly specific rules and overfit. Five keeps nonlinear ability while controlling complexity and readability. `min_samples_split=10` requires at least ten training samples before a node may split. `random_state=42` makes the result reproducible; the number 42 has no special mathematical advantage.

We include it as an interpretable nonlinear model. It has the highest Recall at 77.01%, but its Precision is only 50.35%, so it creates more false alerts. One tree can also be unstable.

---

**Random Forest**

**中文**

Random Forest 建立很多棵略有不同的 Decision Trees，再综合结果。每棵树看到不同 customer samples 和 feature subsets，因此最终判断不会完全依赖一棵树。这种 independent trees + averaging 的方法叫 bagging。

Parameters：

- `n_estimators=200`：森林有 200 棵树。GridSearch 比较 100 和 200，最终 200 的 F1 表现最好。更多树通常提高稳定性，但不保证继续提高准确率。
- `max_depth=12`：每棵树最多约 12 层。GridSearch 比较 8、10、12，最终 12 在当前 regularised range 中表现最好。
- `min_samples_leaf=5`：每个最终 leaf 至少五个 samples。GridSearch 比较 5 和 10，最终 5 在控制过细 leaf 的同时保留较多 pattern。
- `random_state=42`：保证 reproducibility。

为什么调这些值：最初 unrestricted Random Forest 的 train F1=0.999、test F1=0.584，严重 overfitting，因此搜索只使用受限制的 depth 和 leaf size。

结果为最高 Accuracy 75.80%、Precision 53.22%、F1 61.56% 和 AUC 0.839，因此是 overall default。限制是比 Logistic Regression 和单棵树更难解释。

**English**

Random Forest builds many slightly different Decision Trees and combines their results. Different trees see different customer samples and feature subsets, so the final decision does not depend on one tree. This independent-tree averaging method is called bagging.

GridSearch compared 100 and 200 trees, depths 8, 10 and 12, and minimum leaf sizes 5 and 10. The selected settings are 200 trees, maximum depth 12 and minimum leaf size 5 because they produced the best F1 within the regularised search range. `random_state=42` supports reproducibility.

These restrictions were necessary because the original unrestricted forest achieved train F1 of 0.999 but test F1 of only 0.584, showing severe overfitting. The final model has the highest Accuracy, Precision, F1 and AUC, so it is our overall default. Its limitation is lower interpretability.

---

**XGBoost**

**中文**

XGBoost 按顺序建立小树，后面的树重点修正前面留下的错误。这种 sequential correction 叫 boosting。

Parameters：

- `n_estimators=150`：依次加入 150 棵 correction trees；GridSearch 比较 100 和 150。
- `learning_rate=0.1`：每棵树只作较小修正；GridSearch 比较 0.05 和 0.1。太大可能过度修正，太小则需要更多 trees。
- `max_depth=4`：每棵树最多约四层；GridSearch 比较 3 和 4。单棵树保持较浅，因为后面还有很多 correction trees。
- `min_child_weight=5`：新分支要有足够 training information，减少非常小的 branch；GridSearch 比较 5 和 10。
- `subsample=0.8`：每棵树使用约 80% training rows，降低重复依赖完全相同客户。
- `colsample_bytree=0.8`：每棵树使用约 80% features，让不同树关注不同资料。
- `reg_lambda=1`：L2 regularisation，惩罚过度复杂调整；GridSearch 比较 1 和 5。
- `random_state=42`：保证 reproducibility。

为什么选它：提供与 Random Forest 不同的 boosting benchmark。结果为 Accuracy 75.30%、Recall 72.19%、F1 60.81%、AUC 0.830，表现有竞争力，但没有超过 Random Forest。限制是 parameters 较多、需要仔细 tuning，也较难解释。

**English**

XGBoost builds small trees in sequence. Later trees focus on correcting errors left by earlier trees. This sequential correction method is called boosting.

GridSearch compared 100 and 150 trees, learning rates 0.05 and 0.1, depths 3 and 4, child weights 5 and 10, and L2 values 1 and 5. The final model uses 150 trees, learning rate 0.1, depth 4, child weight 5 and L2 value 1. Row and feature subsampling are both 0.8 to reduce repeated dependence on exactly the same data. `random_state=42` supports reproducibility.

We include XGBoost as a boosting benchmark. It achieved F1 of 60.81% and AUC of 0.830. It remains competitive but does not outperform Random Forest. Its limitation is greater tuning complexity and lower interpretability.

## 3. Model Comparison｜模型比较

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.23% | 52.44% | 71.93% | 60.65% | 0.833 |
| Decision Tree | 73.74% | 50.35% | **77.01%** | 60.89% | 0.823 |
| **Random Forest** | **75.80%** | **53.22%** | 72.99% | **61.56%** | **0.839** |
| XGBoost | 75.30% | 52.53% | 72.19% | 60.81% | 0.830 |

**Metric meanings｜指标意思**

- **Accuracy**：所有客户中预测正确的比例｜Correct predictions among all customers.
- **Precision**：所有 churn alerts 中真正 churn 的比例｜Actual churners among all alerts.
- **Recall**：所有实际 churners 中成功找出的比例｜Actual churners detected.
- **F1**：Precision 与 Recall 的平衡｜Balance between Precision and Recall.
- **AUC**：跨 thresholds 的整体排序能力｜Ranking ability across thresholds.

为什么不能只看 Accuracy：73.46% 客户没有 churn；全部预测 No Churn 已可得到约 73.46% Accuracy，但 Recall=0。

Why Accuracy is insufficient: 73.46% of customers did not churn. Predicting No Churn for everyone already gives about 73.46% Accuracy but zero Recall.

---

**Actual mistakes on 1,409 test customers｜在 Test Customers 上实际犯了什么错误**

The stratified test set contains approximately **374 actual churners** and **1,035 retained customers**.

| Model | Caught churners (TP) | Missed churners (FN) | False alerts (FP) | Correct retained (TN) |
|---|---:|---:|---:|---:|
| Logistic Regression | 269 | 105 | 244 | 791 |
| Decision Tree | **288** | **86** | 284 | 751 |
| Random Forest | 273 | 101 | **240** | **795** |
| XGBoost | 270 | 104 | 244 | 791 |

**中文分析**

- Decision Tree 比 Random Forest 多抓到 15 个 churners（288 vs 273），但同时多制造 44 个 false alerts（284 vs 240）。所以它的 Recall 较高，但 Precision 和 Accuracy 较低。
- Random Forest 不是抓到最多 churners，而是在 missed churners 与 false alerts 之间取得更好的整体平衡。
- Logistic Regression 和 XGBoost 的 error counts 非常接近：两者都是约 244 false alerts，分别 miss 105 和 104 个 churners。这说明更复杂的 algorithm 不一定自动产生更好的实际结果。

**English analysis**

- Decision Tree catches 15 more churners than Random Forest, 288 versus 273, but creates 44 more false alerts, 284 versus 240. This explains its higher Recall but lower Precision and Accuracy.
- Random Forest does not catch the most churners. It provides a better overall balance between missed churners and false alerts.
- Logistic Regression and XGBoost have very similar error counts. The more complex algorithm does not automatically produce a better practical result.

**Telco business meaning｜电讯业务意义**

如果 retention team 只能联系有限数量客户，44 个额外 false alerts 代表额外电话、员工时间或不必要 discount。如果每漏掉一个 churner 的收入损失非常高，Decision Tree 多抓到的 15 个 churners 可能更重要。Model choice 需要结合实际 intervention cost，而不只是看排名。

If the retention team has limited capacity, 44 additional false alerts mean extra calls, staff time or unnecessary discounts. If each missed churner causes a very large revenue loss, the 15 additional churners caught by Decision Tree may be more valuable. Model choice therefore depends on intervention cost, not only metric ranking.

---

**How the same Telco customer is processed｜同一个客户在四个 Models 中怎样被处理**

**中文**

假设客户具有以下 profile：刚加入两个月、Month-to-month contract、Fiber optic、Electronic check、月费较高，而且没有 OnlineSecurity 或 TechSupport。

- **Logistic Regression** 把每个 feature 的 coefficient 加在一起。Month-to-month、Electronic check 和高 charge-related values 会增加 score；security/support features 则根据固定 coefficient 影响 score。它的限制是每个 feature 的方向基本固定，较难表达“高月费只在短 tenure 或特定 contract 下特别危险”。`ChargesToTenureRatio` 和 `ContractRiskScore` 是为了帮助 linear model 看见这些结构。
- **Decision Tree** 只让客户沿一条 rule path 前进。例如先根据 contract risk 分开，再根据 tenure 或 charge-related feature 继续分。这个客户可能很快进入高-risk leaf。优点是路径清楚；缺点是接近 split boundary 的两个客户可能被送到完全不同 leaves。
- **Random Forest** 让同一个客户走过 200 棵不同 trees。部分 trees 更重视 tenure，部分重视 contract、charges、payment 或 internet service，最后平均 probabilities。因此一个异常 split 不会完全决定结果。
- **XGBoost** 先产生初始 prediction，再由后续 trees 逐步修正。例如早期 trees 根据 ContractRiskScore 提高风险，后续 trees 再根据 Fiber optic、Electronic check、security 或 dependents 调整剩余 error。

**English**

Consider a customer with two months of tenure, a month-to-month contract, Fiber optic internet, Electronic check, high monthly charges and no OnlineSecurity or TechSupport.

- **Logistic Regression** adds the fitted contribution of every feature. Its feature direction is mostly fixed, so it has difficulty expressing that high charges may be especially risky only for short-tenure or particular contract customers. Engineered features help expose these relationships.
- **Decision Tree** sends the customer through one rule path into one final leaf. The path is readable, but two customers near a split boundary can be sent to very different leaves.
- **Random Forest** sends the customer through 200 different trees. Some trees emphasise tenure, others contract, charges, payment or internet service. Averaging prevents one unusual split from controlling the result.
- **XGBoost** starts with an initial prediction and sequentially corrects the remaining error. Early trees may raise risk using ContractRiskScore, while later trees adjust it using Fiber optic, Electronic check, security or dependent status.

---

**Actual feature reliance in this project｜本项目中实际依赖哪些 Telco Features**

| Random Forest top drivers | Importance | XGBoost top drivers | Importance |
|---|---:|---|---:|
| `ChargesToTenureRatio` | 0.175 | `ContractRiskScore` | 0.283 |
| `ContractRiskScore` | 0.145 | `InternetService_Fiber optic` | 0.147 |
| `tenure` | 0.103 | `PaymentMethod_Electronic check` | 0.089 |
| `MonthlyCharges` | 0.098 | `InternetService_No` | 0.072 |
| `TotalCharges` | 0.082 | `OnlineSecurity` | 0.054 |
| `PaymentMethod_Electronic check` | 0.075 | `ChargesToTenureRatio` | 0.049 |
| `InternetService_Fiber optic` | 0.069 | `Dependents` | 0.035 |
| `OnlineSecurity` | 0.044 | `TechSupport` | 0.030 |

**中文分析**

- Random Forest 的 importance 较分散。它同时使用 customer lifecycle（tenure）、current price pressure（ChargesToTenureRatio、MonthlyCharges、TotalCharges）、contract、payment 和 internet type。因此它较像综合多个 risk signals。
- XGBoost 的 importance 更集中：ContractRiskScore 单独约占 28.3%，Fiber optic 约 14.7%。它更强烈依赖少数 categorical risk signals，再由其他 trees 修正。
- 两个 ensemble 都认为 contract risk、charge-to-tenure relationship、Electronic check、Fiber optic 和 OnlineSecurity 有用。这些共同 drivers 与 EDA 中 month-to-month、Fiber optic、Electronic check 和 early-tenure churn patterns 相呼应。
- Importance 不是 causation，也不能直接比较成“Contract 导致 28.3% churn”。它只表示 fitted model 怎样使用 features。

**English analysis**

- Random Forest spreads importance across customer lifecycle, charge pressure, contract, payment and internet type. It combines a wider set of risk signals.
- XGBoost is more concentrated. ContractRiskScore accounts for about 28.3% of its importance and Fiber optic about 14.7%. It relies strongly on a few categorical risk signals and then applies sequential corrections.
- Both ensembles rely on contract risk, charge-to-tenure relationship, Electronic check, Fiber optic and OnlineSecurity. These shared drivers match the EDA patterns for month-to-month contracts, Fiber optic, Electronic check and early-tenure customers.
- Importance is not causation and does not mean that Contract causes 28.3% of churn.

---

**Why Logistic Regression coefficients look unusual｜为什么 Linear Coefficients 看起来奇怪**

**中文**

Logistic Regression 的最大正 coefficient 包括 MonthlyCharges (+4.796) 和 InternetService_No (+3.246)，较大的负 coefficients 包括 PhoneService (-3.772) 和 Fiber optic (-3.106)。部分方向与简单 EDA 直觉不同，并不代表“没有 internet 会造成 churn”或“Fiber 一定保护客户”。

原因是 MonthlyCharges、internet type、phone/service flags、TotalCharges 和 engineered charge features 高度重叠。Linear model 必须在 correlated predictors 之间分配 weights，所以单一 coefficient 可能不稳定。Logistic Regression 在这里适合当预测 baseline，但不适合把每个 coefficient 当成独立业务 effect。

**English**

Some Logistic Regression coefficients look different from simple EDA intuition. This does not mean that no internet causes churn or that Fiber optic protects customers. MonthlyCharges, internet type, service flags, TotalCharges and engineered charge features overlap strongly. The linear model must distribute weights among correlated predictors, making individual coefficients unstable. Logistic Regression remains a useful predictive baseline, but its coefficients should not be treated as independent business effects.

---

**Direct project-specific differences｜直接对应本项目的差异**

| Question | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| How does it combine telco signals? | Adds fixed feature contributions | Uses one customer rule path | Averages many different rule paths | Sequentially corrects earlier errors |
| How does it handle interactions? | Mainly through engineered features | Explicitly through nested splits | Many interaction patterns across trees | Interaction corrections accumulated over trees |
| What does it emphasise here? | Charge/service coefficients, but VIF-affected | High-Recall rules | Broad lifecycle + charge + contract balance | Concentrated contract + Fiber + payment signals |
| Main test-set behaviour | Competitive simple baseline | Catches most churners, most false alerts | Fewest false alerts and best balance | Similar errors to Logistic despite more complexity |
| Best use in this project | Transparent benchmark | When missed churn is most costly | Default retention prioritisation | Boosting benchmark / future tuning |

---

**Which metric matters most?｜哪个指标最重要？**

**中文**

Recall 很重要，因为 false negative 代表漏掉真正 churner。但只追求 Recall 会产生很多 false positives，因此本项目用 F1 作为主要平衡指标，并用 AUC 检查跨 thresholds 的 ranking ability。

**English**

Recall is important because a false negative means missing an actual churner. However, maximising Recall creates false positives, so we use F1 as the main balancing metric and AUC for ranking ability across thresholds.

---

**Which model is best?｜哪个模型最好？**

**中文**

Random Forest 是 overall default，因为 Accuracy、Precision、F1 和 AUC 最高，同时保持约 73% Recall。Decision Tree 如果业务唯一目标是抓到最多 churners，则 Recall 77.01% 更高。不能说 Random Forest 在所有业务情况下都最好。

**English**

Random Forest is our overall default because it has the highest Accuracy, Precision, F1 and AUC while maintaining about 73% Recall. Decision Tree may be preferred if the only objective is catching the most churners because its Recall is 77.01%. Random Forest is not universally best for every business objective.

Metric gaps are small:

- Random Forest vs Logistic Regression: F1 improves by only **0.91 percentage points** and AUC by **0.006**.
- Random Forest vs XGBoost: F1 improves by only **0.75 percentage points** and AUC by **0.009**.
- Random Forest vs Decision Tree: Recall is **4.02 points lower**, but Precision is **2.87 points higher** and Accuracy **2.06 points higher**.

McNemar's paired test found a statistically significant difference only between Decision Tree and Random Forest (`p≈0.012`). Random Forest was not significantly different from Logistic Regression or XGBoost at the 5% level. Therefore, we say it leads the observed point estimates, not that it is conclusively superior to every model.

---

**Why did XGBoost not beat Random Forest?｜为什么 XGBoost 没有胜过 Random Forest？**

**中文**

更复杂不代表一定更好。本项目只有约 7,000 rows，而且 engineered features 已经直接暴露 ContractRiskScore 和 ChargesToTenureRatio 等强 signals。Random Forest 已能稳定捕捉大部分 nonlinear patterns。XGBoost 的 tuning grid 也刻意保持 regularised，并没有无限扩大搜索。结果显示它与 Logistic Regression 的实际 error counts 很接近，而不是明显超过其他模型。

**English**

Greater complexity does not guarantee better performance. This project has about 7,000 rows, and engineered features already expose strong signals such as ContractRiskScore and ChargesToTenureRatio. Random Forest can capture most nonlinear patterns reliably. The XGBoost search grid is also intentionally regularised rather than unlimited. Its actual error counts are close to Logistic Regression rather than clearly better.

---

**Which model matches which retention situation?｜不同 Retention 情况选择什么 Model？**

| Business situation | Recommended model | Project-specific reason |
|---|---|---|
| Retention team has limited call capacity | Random Forest | Fewest false alerts: 240 |
| Missing a churner is extremely expensive | Decision Tree | Fewest missed churners: 86 |
| Management requires simple coefficient/rule explanation | Logistic Regression or Decision Tree | More transparent model structure |
| Need a boosted nonlinear benchmark | XGBoost | Sequential correction and concentrated risk drivers |
| Need the default overall balance | Random Forest | Best Accuracy, Precision, F1 and AUC point estimates |

---

**Could another model be added?｜可以加入什么新模型？**

**中文**

可以加入 HistGradientBoosting，作为 scikit-learn-native boosting benchmark。它用 histogram bins 加快训练，可以和 XGBoost 比较。新增模型仍必须使用同样 split、fold-level resampling 和 metrics，不能因为多一个模型就改变比较规则。

**English**

We could add HistGradientBoosting as a scikit-learn-native boosting benchmark. It uses histogram bins for efficient training and provides a useful comparison with XGBoost. It must still use the same split, fold-level resampling and evaluation metrics.

---

**Fast shared answers｜共同快速回答**

- Why not Accuracy alone? → A no-churn prediction already gets about 73.5% Accuracy but zero Recall.
- DT vs RF? → One readable tree versus many averaged trees that reduce instability.
- RF vs XGBoost? → Independent bagging versus sequential error-correcting boosting.
- Does importance prove causation? → No. It shows model reliance, not what causes churn.
- Does one higher customer probability identify the best model? → No. Model quality is measured across the common labelled test set.

## 4. Codebase & Model Operation｜代码结构与模型操作

**What this section covers｜这一部分说明什么**

**中文**

这一份 HTML 只解释数据如何进入模型、parameters 在哪里设置、怎样 tuning、怎样重新训练、怎样产生 prediction，以及哪些文件负责这些操作。Streamlit 页面展示会放在另一份独立 HTML。

**English**

This HTML explains how data enters the models, where parameters are set, how tuning and retraining work, how predictions are produced, and which files control these operations. Streamlit page presentation will be documented in a separate HTML guide.

---

**Main code files｜主要代码文件**

| File | 中文用途 | English purpose |
|---|---|---|
| `src/data_prep.py` | 读取、清理、engineer、encode，建立 23-feature matrix | Load, clean, engineer and encode the 23-feature matrix |
| `src/modelling.py` | Split、scale、SMOTE、训练与 GridSearch | Split, scale, resample, train and tune models |
| `src/evaluation.py` | Metrics、K-fold、importance、threshold、McNemar test | Metrics, K-fold, importance, thresholds and McNemar test |
| `src/eda.py` | Statistical tests、VIF、feature validation | Statistical tests, VIF and feature validation |
| `streamlit/train_model.py` | 用最终固定 parameters 训练 deployment models 并保存 artefacts | Train deployment models with final fixed parameters and save artefacts |
| `streamlit/app.py` | 加载 artefacts、转换新客户、调用 prediction | Load artefacts, transform a customer and call prediction |
| `notebooks/analysis.ipynb` | 按报告顺序运行完整 analysis | Run the full analysis in report order |

---

**Preprocessing order｜模型看到数据之前发生什么**

```text
Raw CSV
→ TotalCharges converted to numeric
→ tenure-zero blanks filled with 0
→ engineered features created
→ redundant service categories collapsed
→ target Churn converted to 0/1
→ categorical values encoded
→ final 23 columns aligned
→ stratified train-test split
→ scaler fitted on training numeric columns
→ training data resampled with SMOTE
→ models fitted
```

**中文**

顺序很重要。Scaler 只能从 training data 学习 mean 和 standard deviation；SMOTE 也只能应用在 training data。Test data 只能使用已经 fitted 的 scaler，不能参与 fitting 或 resampling。

**English**

The order matters. The scaler may learn means and standard deviations only from training data. SMOTE may also be applied only to training data. Test data uses the already fitted scaler and must not participate in fitting or resampling.

---

**Which numerical columns are scaled?｜哪些数值 Features 被 Scaling？**

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `ChargesToTenureRatio`

**中文**

Scaling 把不同单位转换到相近 scale。Logistic Regression 对此最需要；tree models 不依靠数值距离，所以通常不需要 scaling，但本项目为了让四个模型接收同一 matrix，仍提供相同 scaled inputs。

**English**

Scaling places different numerical units on comparable scales. Logistic Regression benefits most from it. Tree models do not depend on numerical distance and normally do not require scaling, but this project supplies the same scaled matrix to all four models for a consistent pipeline.

---

**Actual model constructors｜代码实际建立的 Models**

```python
LogisticRegression(
    max_iter=2000,
    random_state=42,
)

DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42,
)

RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
)

XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42,
    eval_metric="logloss",
)
```

---

**Explicit settings vs library defaults｜明确设置与默认设置**

**中文**

只有上面 constructor 中写出的 parameters 是项目明确指定的。没有写出的 parameters 使用 library defaults。组员不要声称所有 defaults 都经过 tuning。

**English**

Only the parameters written in the constructors are explicitly specified by the project. Unwritten parameters use library defaults. We should not claim that every default was tuned.

| Model | Important defaults currently relied on | 中文说明 |
|---|---|---|
| Logistic Regression | L2 penalty, `C=1.0`, `solver="lbfgs"`, no class weight | 使用 sklearn defaults；没有另外 tuning |
| Decision Tree | Gini criterion, best splitter, `min_samples_leaf=1`, no class weight | 单棵树只明确限制 depth 和 split size |
| Random Forest | Gini criterion, bootstrap enabled, `max_features="sqrt"`, no class weight | 每棵树使用 bootstrap sample 和随机 feature subset |
| XGBoost | tree booster, binary logistic objective, no extra positive-class weight | 依靠 SMOTE，不再另外使用 `scale_pos_weight` |

为什么没有 `class_weight` 或 `scale_pos_weight`：本项目已经使用 SMOTE 平衡 training information；再加入 algorithmic class weighting 可能 double-correct minority class。

Why no `class_weight` or `scale_pos_weight`: the project already uses SMOTE to balance training information. Adding algorithmic class weighting could double-correct the minority class.

---

**Which models are tuned?｜哪些 Models 经过 GridSearch？**

| Model | Selection method | 中文说明 |
|---|---|---|
| Logistic Regression | Fixed baseline | 只设 `max_iter=2000`，没有 GridSearch |
| Decision Tree | Fixed shallow configuration | depth 5、split 10 是为了 readability 与 complexity control，不是 GridSearch winner |
| Random Forest | `GridSearchCV`, `cv=5`, scoring=`f1` | 从 regularised grid 选 final estimator |
| XGBoost | `GridSearchCV`, `cv=5`, scoring=`f1` | 从 boosting grid 选 final estimator |

**Random Forest tuning grid**

```python
{
    "n_estimators": [100, 200],
    "max_depth": [8, 10, 12],
    "min_samples_leaf": [5, 10],
}
```

总共有 2 × 3 × 2 = **12 parameter combinations**。每组进行 five-fold CV，所以需要 12 × 5 = **60 model fits**，不包括 GridSearch 最后用最佳 settings 重新 fit 的过程。

**XGBoost tuning grid**

```python
{
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 150],
    "min_child_weight": [5, 10],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1, 5],
}
```

总共有 2 × 2 × 2 × 2 × 1 × 1 × 2 = **32 parameter combinations**。Five-fold CV 需要 32 × 5 = **160 model fits**，不包括最后 refit。

**中文**

GridSearch 不是随便尝试一个 parameter。它训练每一种组合，用五折平均 F1 比较，再选出表现最好的组合。选择 F1 是因为 churn imbalanced，并且我们需要平衡 missed churners 与 false alerts。

**English**

GridSearch does not try only one parameter. It trains every combination, compares the mean five-fold F1 and selects the strongest combination. F1 is used because churn is imbalanced and we need to balance missed churners with false alerts.

---

**Analysis/tuning path vs deployment path｜分析调参与部署训练的区别**

**中文**

`src/modelling.py` 是 analysis/tuning path。它可以重新运行 GridSearch 来寻找 Random Forest 和 XGBoost settings。

`streamlit/train_model.py` 是 deployment artefact path。它不在每次启动时重新 GridSearch，而是直接使用已经选定的 final parameters，训练四个 models 并保存 files。这样 Streamlit 启动更快、结果也固定。

**English**

`src/modelling.py` is the analysis and tuning path. It can rerun GridSearch to find Random Forest and XGBoost settings.

`streamlit/train_model.py` is the deployment-artifact path. It does not rerun GridSearch every time. It uses the selected final parameters, trains all four models and saves the required files. This makes application startup faster and keeps results fixed.

---

**Saved model artefacts｜训练后保存什么**

| File | Contents | 为什么需要｜Why needed |
|---|---|---|
| `models.pkl` | Four fitted model objects | 不必每次启动重新训练｜Avoid retraining on startup |
| `scaler.pkl` | Fitted StandardScaler | 新客户必须使用相同 training scale |
| `feature_columns.pkl` | Ordered list of 23 features | 保证新客户 column 名称和顺序一致 |

**中文**

三个 files 必须来自同一次 training run。如果只替换 models.pkl，却保留旧 scaler 或旧 feature order，prediction 可能错误或直接失败。

**English**

All three files must come from the same training run. Replacing only models.pkl while keeping an old scaler or feature order may produce incorrect predictions or cause failure.

---

**How a prediction is produced｜代码怎样产生 Prediction**

```text
One customer profile
→ apply consistency rules
→ clean and engineer features
→ encode to 23 columns
→ reindex to saved feature order
→ transform four numeric columns with saved scaler
→ model.predict_proba(...)[0, 1]
→ churn risk score
→ apply threshold if a Yes/No class is needed
```

**中文**

`predict_proba` 返回两个 probabilities：class 0（retained）与 class 1（churn）。代码使用 `[0, 1]` 取得第一个客户的 churn probability。`model.predict()` 通常根据默认 0.50 threshold 直接产生 0/1；`predict_proba()` 保留完整 risk score，适合 threshold analysis。

**English**

`predict_proba` returns probabilities for class 0, retained, and class 1, churn. `[0, 1]` selects the first customer's churn probability. `model.predict()` normally produces a 0/1 result using the default 0.50 threshold, while `predict_proba()` preserves the full risk score for threshold analysis.

---

**How to retrain deployment artefacts｜怎样重新训练 Models**

From the project root:

```powershell
cd streamlit
python train_model.py
```

Expected outputs:

```text
streamlit/models.pkl
streamlit/scaler.pkl
streamlit/feature_columns.pkl
```

**中文**

只有当 dataset、features、preprocessing、model parameters 或 library versions 改变时，才需要重新训练 artefacts。重新训练后必须再次检查 metrics，不能假设旧的 displayed numbers 仍然正确。

**English**

Retrain the artefacts when the dataset, features, preprocessing, model parameters or library versions change. After retraining, all metrics must be checked again; the old displayed numbers cannot be assumed to remain correct.

---

**How models are evaluated｜怎样计算表现**

`src/evaluation.py` provides:

- `evaluate_model()` → Accuracy, Precision, Recall, F1 and AUC on test data;
- `evaluate_all_models()` → same metrics for all four models;
- `kfold_cv_all_models()` → stratified five-fold mean, standard deviation and fold scores;
- `overfitting_check()` → train F1 vs test F1;
- `mcnemar_test()` → paired correctness comparison on the same test customers;
- `optimal_threshold()` → exploratory threshold with strongest test-set F1.

**中文**

Hold-out test metrics 和 K-fold metrics 回答不同问题。Test metrics 是 final unseen-set point estimate；K-fold 检查 training partition 内不同 folds 的平均表现与稳定性。两者数值不必完全一样。

**English**

Hold-out test metrics and K-fold metrics answer different questions. Test metrics provide a final unseen-set point estimate. K-fold examines mean performance and stability across training-partition folds. Their values do not need to be identical.

---

**Important implementation cautions｜需要诚实说明的代码限制**

1. **Tuning SMOTE context**

   - 中文：当前 `src/modelling.py` 的 full pipeline 先对整个 training split 做 SMOTE，再把 resampled data 交给 GridSearchCV。更严格的方法应把 SMOTE 放进 GridSearch pipeline，使每个 fold 只根据自己的 training portion 建立 synthetic samples。
   - English: The current full tuning path applies SMOTE to the complete training split before GridSearchCV. A stricter implementation should place SMOTE inside the GridSearch pipeline so that synthetic samples are created only from each fold's training portion.

2. **Displayed reliability CV**

   - 中文：Streamlit reliability CV 已把 SMOTE 放在每个 fold 内，但 scaling 是在 fold 切分前根据 outer training partition fitted。更严格的版本应把 scaling、SMOTE 和 model 放进同一个 fold-level pipeline。
   - English: The displayed reliability CV places SMOTE inside each fold, but scaling is fitted on the outer training partition before the fold split. A stricter version should place scaling, SMOTE and the model in one fold-level pipeline.

3. **Hard-coded displayed test metrics**

   - 中文：`streamlit/app.py` 保存一份已 cross-check 的 test metric display cache。重新训练 models 后必须同步重新计算和更新，否则页面可能显示旧数值。
   - English: `streamlit/app.py` contains a cross-checked display cache of test metrics. After retraining, the metrics must be recalculated and updated or the page may show old values.

4. **Default parameters**

   - 中文：没有写入 constructor 的 settings 使用 library defaults。Library version 改变时，应重新检查 defaults 与 saved pickle compatibility。
   - English: Settings not written in the constructor use library defaults. When library versions change, defaults and saved-pickle compatibility should be checked again.

**Safe statement for presentation｜安全说法**

> Our displayed five-fold stability evaluation applies SMOTE separately inside each training fold. A future codebase improvement is to place scaling, SMOTE and model fitting inside one complete pipeline for both tuning and evaluation.
