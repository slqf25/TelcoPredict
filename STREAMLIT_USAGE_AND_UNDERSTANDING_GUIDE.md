# Streamlit System Guide

这份指南按照实际展示系统的顺序编排：先预测一位客户，再说明数据依据，最后比较模型和业务取舍。组员不需要背下所有数字，但必须理解每个页面为什么存在，以及它怎样连接到我们的 Telco Churn project。  
This guide follows the actual demonstration flow: predict one customer, show the supporting data evidence, then compare models and operational trade-offs. Team members do not need to memorise every number, but they should understand why each page exists and how it connects to our Telco Churn project.

```text
Predict customer risk
        ↓
Understand the data and feature decisions
        ↓
Compare model performance and reliability
        ↓
Choose a threshold based on retention priorities
```

---

## 1. Predict Workspace / 客户预测页面

### 1.1 从客户资料到预测结果 / From customer profile to prediction

Predict 页面解决一个实际问题：**如果公司现在收到一位客户的资料，系统怎样判断他是否值得优先进行 retention follow-up？**  
The Predict workspace answers one operational question: **Given a current customer profile, should this customer be prioritised for retention follow-up?**

页面顶部先选择 active model。这里的 Logistic Regression、Decision Tree、Random Forest 和 XGBoost 是四种不同的 **预测方法**。系统默认使用 Random Forest；点击另一个 model 后，同一位客户会交给另一个 fitted model 计算。顶部五个数值是该模型在全部 1,409 位 test customers 上的表现，不是当前客户的结果；五个数值的意思会在 Models 页面解释。

The four active-model buttons select four different **prediction methods**. The top five numbers evaluate the selected model across the complete test set; they are not five additional models and do not describe the current customer.

#### 为什么系统提供三个 Demo Presets？ / Why do we provide three presets?

Presets 的目的不是为了让页面看起来方便，而是用三个有明确用途的 profiles 快速检查系统行为：明显高风险、明显低风险和普通参考案例。

The presets provide three purposeful system checks: a clearly high-risk profile, a clearly low-risk profile and a typical reference profile.

| Preset | 为什么建立这个组合 / Why this profile exists | 展示时预期看到什么 / Expected demonstration |
|---|---|---|
| **Highest-risk segment** | 把 EDA 中较高 churn 的真实 patterns 放在一起：0–12 month tenure churn 47.4%、month-to-month 42.7%、electronic check 45.3%、fiber optic 41.9%，再配合高 monthly charge 和没有 security/support add-ons。它用于检查模型是否会把明显危险组合放到优先名单。 / Combines observed higher-risk patterns to test whether the model responds strongly. | 应得到明显高于 26.5% base rate 的结果；截图中 RF 给出 93.0%。 / A score clearly above the base rate; RF returns 93.0% here. |
| **Lowest-risk segment** | 使用数据中较稳定的 patterns：long tenure、two-year contract、automatic payment、partner/dependents 和多项 services。它用于检查系统是否并非把所有客户都预测成 high risk。 / Uses more stable patterns as a sanity check that the system does not flag everyone. | 应明显低于 high-risk preset。 / A substantially lower score than the high-risk profile. |
| **Dataset average** | 使用接近常见类别和平均 tenure/monthly charge 的输入。它不是“平均人的真实记录”，而是 high 与 low 之间的参考点。 / Provides a typical reference profile rather than an actual average person. | 通常位于两种极端案例之间，帮助说明模型不是只有 high/low 两种输出。 / Usually falls between the two extremes. |

这些 presets 是根据 dataset patterns 人工组合的 demonstration profiles，不是数据集中三位真实客户，也不保证是数学上绝对最高或最低的 possible prediction。

These are constructed demonstration profiles, not exact customer records or guaranteed mathematical extremes.

客户资料在画面上分成六组：Demographics、Charges、Contract、Billing、Core Services 和 Add-ons。中间的 3D tower 只是把这些 inputs 组织成 telecom-themed interface；它本身不计算 risk。

The profile is organised into six user-facing groups. The 3D tower is only a telecom-themed input interface; it does not calculate risk.

系统真正执行的是：

```text
Six input groups
→ check service consistency
→ engineer two extra features
→ one-hot encode categories
→ scale four numeric columns
→ align the same 23 columns used in training
→ selected_model.predict_proba()
```

六组 UI inputs 最后成为 **23 个 model predictors**。两个不是由用户直接输入的 engineered features 是：

- `ContractRiskScore`：Month-to-month = 2、One year = 1、Two year = 0。它把 contract commitment 的顺序变成模型可用的数值，不代表数值 2 的客户一定 churn。
- `ChargesToTenureRatio = MonthlyCharges / (tenure + 1)`：比较 monthly charge 与客户资历。`+1` 避免 tenure 为 0 时除以 0。

The interface also creates `ContractRiskScore` and `ChargesToTenureRatio`. These are model inputs learned from historical patterns, not causal rules.

数值 predictors 中的 tenure、MonthlyCharges、TotalCharges 和 ChargesToTenureRatio 会使用训练时保存的 StandardScaler。Scaling 只是把不同单位转换到模型熟悉的尺度，特别帮助 Logistic Regression optimisation；不会凭空改变客户风险。

The four numeric predictors are transformed using the scaler fitted during training. Scaling changes representation, not the customer's underlying information.

系统还会处理不一致输入：如果 Internet Service = No，internet add-ons 会按照训练数据中的 no-internet representation 送入模型；如果 Phone Service = No，Multiple Lines 会按 no-phone representation 处理。Total Charges 的 Auto mode 使用 `tenure × monthly charge` 估算，因此有真实 billing history 时应使用真实累计费用。

The system applies training-consistent service representations. Auto Total Charges is only an estimate and should be replaced with actual billing history when available.

展开 `Preview how this profile becomes model input` 可以直接查看 UI selections 怎样变成 23 columns。DSL 和 Bank transfer (automatic) 是 one-hot reference categories；没有独立 column 不代表它们被忽略，而是当同组其他 dummy columns 都为 0 时代表 reference category。

The preview is the clearest evidence of how human-readable inputs become model inputs. Reference categories are represented by the absence of the other dummy indicators.

### 1.2 怎样解释 93% 的 High Risk 结果 / How to explain the 93% result

截图使用 Highest-risk preset：2 months tenure、month-to-month、fiber optic、electronic check、monthly charge $95，而且没有 internet add-ons。Random Forest 输出 **93.0% churn probability**。

For this high-risk preset, Random Forest estimates a **93.0% churn probability**.

#### 先看结果第一排 / Read the result row first

| 画面内容 / Display | 它到底是什么 / What it actually is | 结合当前结果怎样说 / How to interpret this result |
|---|---|---|
| **HIGH RISK · 93.0%** | Random Forest 对当前这组 23 predictors 输出的 churn probability。它不是模型 Accuracy。 / RF probability for this profile, not model Accuracy. | 模型认为这位客户比一般客户危险很多，应优先 review；但 93% 不是保证他一定 churn。 / Prioritise the customer, but do not treat the estimate as certainty. |
| **Random Forest** | 告诉我们 93% 是由哪个 active model 算出来的。 / Identifies the model that produced 93%. | 如果切换成 LR、DT 或 XGB，同一 profile 会得到另一个 probability。 |
| **Dataset base rate 26.5%** | Dataset 中 7,043 位客户有 1,869 位 churn：`1,869 ÷ 7,043 = 26.5%`。它回答“在还没看这位客户资料前，历史数据中一般客户有多常 churn？” / Historical background churn frequency before considering this profile. | 93.0% 明显高于一般背景的 26.5%，所以这个 profile 不是普通风险。26.5% 不是 threshold，也不是这位客户的第二个预测。 |
| **+66.5 pp vs base rate** | `93.0% − 26.5% = 66.5 percentage points`。`pp` 是 percentage points，不是 66.5% increase。 | 当前 prediction 比历史平均风险高 66.5 个百分点。 |
| **0/5 retention signal** | UI 把 churn probability 反向压缩成五格视觉提示；risk 越高，retention bars 越少。它不是模型训练出来的新 metric。 / A simplified inverse visual, not another metric. | 93% churn risk 很高，所以只剩 0/5；展示时不需要用它进行 model comparison。 |

一句完整说法：

> Random Forest estimates 93.0% churn risk for this profile. The historical dataset base rate is 26.5%, meaning 1,869 of 7,043 customers churned. This customer's estimate is therefore 66.5 percentage points above the background rate and should be prioritised, although it is not a guaranteed outcome.

#### 再看横向 risk graph / Then read the horizontal risk graph

| Graph element | 怎样看 / How to read it |
|---|---|
| Green 0–40 | UI 标为 Stable / Low Risk。 |
| Amber 40–70 | UI 标为 At Risk / Medium Risk。 |
| Red 70–100 | UI 标为 Critical / High Risk。 |
| Grey mark at 26.5 | Dataset base rate 的位置。 |
| Diamond at 93.0 | 当前客户的 prediction；越靠右代表 predicted churn risk 越高。 |

40% 与 70% 是人为选择的 **display bands**，不是 GridSearch 找到的参数，也不是业务验证后的最佳 decision threshold。Models 页的 0.50 slider 才是在研究 probability 怎样转成 Yes/No decision。

The 40% and 70% boundaries organise the display; they are not learned or validated operational thresholds.

#### 51.4% evidence box 怎样和 93% 一起读？

51.4% 是历史数据中“0–12 months + month-to-month”这一群客户的实际 churn rate，只使用两个条件；93% 是 Random Forest 使用当前客户全部 23 predictors 的个人预测。Current profile 还有 fiber optic、electronic check、high charge 和 no add-ons 等 signals，所以两个数值不需要相同。

The 51.4% value is a two-variable group rate; 93% is a full-profile model estimate. They answer different questions.

#### Model-wide important features graph 怎样看？

这张图回答：**Random Forest 在所有客户上作判断时，整体最常依赖哪些 business information？** 它不是专门拆解当前 93% 的 local explanation。

| Graph element | 当前画面 / Current screen | 正确解释 / Correct interpretation |
|---|---|---|
| Bar position | 从左到右列出 top six business-level features。 | 只显示最重要的六组，避免 23 columns 全部挤在 Predict 页面。 |
| Bar height | Charges/tenure 17.5%、Contract 14.6%、Tenure 10.3%、Monthly charges 9.8%、Internet 9.3%、Payment 9.0%。 | 百分比是 RF global split importance 重新归一化后的相对份额；越高表示整个 forest 越依赖它。 |
| Orange bars | 排名前两位。 | 只是视觉强调，不代表它们一定把当前客户 risk 往上推。 |
| Customer value below bar | 例如 2 months、Month-to-month、$95、Fiber optic、Electronic check。 | 帮我们把全局重要 feature 联系到当前 profile；不是该 feature 对 93% 的贡献值。 |

与当前案例的联系是：RF 平常最依赖的几组 information，恰好都是这个 high-risk profile 的核心资料；而 EDA 也显示 short tenure、month-to-month、fiber optic 和 electronic check 有较高 observed churn。因此 93% 有合理的 data context。**但这张 importance graph 本身不能告诉我们每一项增加了多少 risk，也不能证明 causation。**

The chart shows that the current profile contains values from feature groups the forest relies on heavily, but it does not quantify local contributions or causal effects.

图中的 `Monthly charge / (tenure + 1)` 现在直接使用与模型相同的公式。对于当前客户，它显示 `95 ÷ (2 + 1) = 31.67 model value`，因此图表 label、preview 和实际 prediction input 保持一致。

#### Model consensus graph 怎样看？

这张图回答：**如果同一位客户交给四个模型，它们是否得出相似判断？**

| Graph element | 怎样看 / How to read it |
|---|---|
| Four rows | 四个 active models，不是四个 metrics。 |
| Dot position | 每个模型对同一 profile 输出的 churn probability；越右越高。 |
| Star | 当前选择的 Random Forest。 |
| Background colours | 与上方相同的 Low、Medium、High display bands。 |
| 6.9 percentage-point spread | 最高 LR 99.9% 减最低 RF 93.0%，表示四个 predictions 相差 6.9 pp。 |

当前四个结果是 LR 99.9%、DT 94.6%、RF 93.0%、XGB 97.4%，全部落在 High Risk band。正确结论是：**虽然算法不同，四个模型都认为这个 profile 应优先关注，而且分歧只有 6.9 pp。** 但它们使用相同 dataset 和 features，所以 agreement 不是客户一定 churn 的证明。

All four algorithms agree on the broad high-risk decision; consensus shows consistency, not guaranteed correctness.

### 1.3 Predict 页面现场说法 / Predict-page presentation script

建议用约一分钟完成：

1. 选择 Random Forest：`Random Forest is our default because it gives the best overall F1 and AUC balance.`
2. 点击 Highest-risk segment，指出 short tenure、month-to-month、fiber optic、electronic check、high charge 和 no add-ons。
3. 说明：`The tower only organises the inputs. The system converts the profile into the same 23 predictors used during training.`
4. 点击 Predict：`The model estimates 93% churn risk. This is a probability estimate, not certainty.`
5. 指出 base rate 与 51.4% segment evidence 的区别。
6. 结束时说：`Global importance shows model reliance, while consensus shows whether the four models broadly agree.`

| Question | Short answer |
|---|---|
| Why is 93% different from 51.4%? | 51.4% uses only tenure band and contract; 93% uses all 23 predictors. |
| Does 93% mean certain churn? | No. It is a probability estimate used for prioritisation. |
| Why use 40% and 70%? | They are simple UI risk bands, not trained or validated operational thresholds. |
| Does the 3D tower affect prediction? | No. It is only the input interface. |
| Are important features explaining this customer? | No. They are global model importance, not local contribution. |
| Do all four models agreeing prove the result? | No. They share the same data and may share errors. |

---

## 2. Data Analysis Workspace / 数据与 Feature 依据

### 2.1 这个页面怎样支持系统 / How this page supports the system

老师不会重点检查 EDA，所以现场不需要逐张讲图。这个 workspace 的作用是证明：**Predict 页面使用的 inputs 和 Models 页面的结论不是凭空选择，而是经过清理、统计检查和 feature audit。**

Detailed EDA is not the presentation focus. This workspace exists to make the preprocessing and feature decisions traceable.

| System fact | Why it matters |
|---|---|
| 7,043 customers；1,869 churned，5,174 retained | Churn 只有 26.5%，所以是 minority class，不能只看 Accuracy。 |
| 21 raw columns → 23 model predictors | Categories 被编码，两个 engineered features 被保留。 |
| 5,634 train；1,409 test | 80/20 stratified split 保持相近 class proportion。 |
| 11 blank TotalCharges values | 转成 numeric 后才被发现并处理；没有删除这 11 位客户。 |
| Scaling 与 SMOTE 只使用 training observations | 保护 held-out test set，避免 test information 进入 training。 |

主要 observed patterns 是 month-to-month churn 42.7%、first-year churn 47.4%、electronic-check churn 45.3%，都高于 26.5% baseline；fiber optic customers 为 41.9%。这些 patterns 帮助模型找到 priority segments，但都是 **association，不是 causation**。

The main observed high-risk patterns are month-to-month contracts, first-year tenure, electronic checks and fibre optic service. They support prioritisation but do not prove causal effects.

Heatmap 只需知道：每格是某个 customer combination 的 churn rate 和 sample size。颜色深表示 observed churn 较高，但小样本格不能过度解释。例如 month-to-month + manual payment 是 46.5%，而 two-year customers 的 manual 与 automatic payment 几乎相同；这表示 payment pattern 会随 contract context 改变，却不能证明强制转 autopay 会减少 churn。

Heatmaps expose conditional patterns. Any proposed intervention still requires a controlled test.

### 2.2 Feature decisions 与 VIF / Feature decisions and VIF

八个 candidate engineered features 中，最终只把两个送入模型：

| Kept feature | 为什么保留 / Why kept |
|---|---|
| ContractRiskScore | 用 2、1、0 表示 contract commitment，并替代 raw Contract，避免同时保留重复 representation。 |
| ChargesToTenureRatio | 捕捉 monthly charge 相对于 tenure 的关系，在分析中与 churn 有较强 association。 |

其他 engineered features 没进入模型不是因为它们“完全没用”，而是因为与现有 predictors 重复或只适合 EDA：TenureGroup 重复 continuous tenure；TotalServicesSubscribed 是 service flags 的总和；IsAutoPay 与 PaymentMethod 重叠；AvgChargePerMonth 接近 MonthlyCharges；HasInternetService 会隐藏 Fiber/DSL 差别；HasPartnerOrDependents 会损失两个原始 fields 的细节。

The other engineered features were excluded mainly because they duplicated existing information or were more useful for EDA than modelling.

如果老师问“当前 23 个 predictors 中哪个可以进一步 remove”，第一候选是 **gender**：它的 Cramer's V 只有 0.008，而且 p-value 0.487，没有明显单变量 churn association。PhoneService 也很弱，但会牵涉 service encoding semantics，所以 gender 是更干净的第一项 ablation candidate。

Among the current predictors, **gender** is the first removal candidate because it shows almost no univariate association. The proper decision still requires an ablation test: remove it, retrain all models, and compare held-out F1, Recall, AUC and stability.

VIF 检查的不是 feature 与 churn 的关系，而是某个 predictor 能否被其他 predictors 大量推测出来。当前 Streamlit 对完整 encoded matrix 的计算显示 MonthlyCharges VIF = **866.1**，并有 7 个 predictors 高于 10。这表示 charge、internet 和 service variables 有严重 information overlap。

VIF measures overlap among predictors, not predictive relationship with churn. The current result indicates severe overlap among charge and service variables.

866.1 不代表与 churn 有 866 倍关系，也不代表 Accuracy 下降 866%。当前函数没有为 VIF matrix 加 intercept，因此 exact magnitude 对 encoding 很敏感；应该重视“非常高”的结论，而不是赋予 866.1 商业意义。

The exact VIF magnitude is representation-sensitive. It should be treated as a multicollinearity warning, not a business effect size.

为什么不直接删除 MonthlyCharges？因为它仍是 Random Forest 的重要 predictor。高 VIF 最直接影响 Logistic Regression coefficients 的独立解释；tree models 通常较能容忍 correlated predictors，但 importance 可能被相关 features 分摊。正确做法是 ablation testing，不是根据一个 VIF 自动删除。

High VIF mainly limits independent coefficient interpretation. Any removal should be validated by retraining and comparing model performance.

这一页现场只需一句话：

> The Data Analysis workspace documents how 7,043 customer records became 23 audited predictors. It supports our feature choices, but observed associations and importance values are not treated as causal evidence.

---

## 3. Models Workspace / 模型比较与业务决策

### 3.1 Performance 与实际 model choice / Performance and actual model choice

先分清楚三种东西，否则 Model tab 会看起来像有九个 models：

| 名称 / Name | 它是什么 / What it is | 我们系统里的例子 / Example in our system |
|---|---|---|
| **Model** | 学习历史数据并对客户输出 probability 的方法。 / A method that learns patterns and produces probabilities. | Logistic Regression、Decision Tree、Random Forest、XGBoost——一共四个。 |
| **Metric** | 用来评价 model predictions 的“尺”，本身不会做预测。 / A ruler used to score predictions; it does not predict customers. | Accuracy、Precision、Recall、F1、AUC——一共五个。 |
| **Evaluation method** | 决定在哪里、怎样用 metrics 检查模型。 / The procedure used to evaluate models. | 1,409-customer held-out test set、5-fold cross-validation。 |

系统关系是：

```text
Customer predictors
→ one of four MODELS
→ churn probability
→ threshold converts probability to Yes/No
→ compare Yes/No with actual outcome
→ calculate five METRICS
```

以下用 Random Forest 在 threshold 0.50 的真实结果说明五个 metrics。Test set 有 1,409 位客户：TN 795、FP 240、FN 101、TP 273。老师要求的 summary 应该包含 **数字代表的客户、对 retention operation 的意义，以及这个 metric 没告诉我们的东西**。

| Metric | 当前数字与客户人数 / Number and customers | 与 Telco churn 的实际关系 / Telco interpretation | Presentation summary — 不要只念数字 |
|---|---|---|---|
| **Accuracy 75.8%** | `(795 TN + 273 TP) ÷ 1,409`，即 1,409 位客户中有 1,068 位整体分类正确。 | 它同时计算正确 retained 和正确 churn，但 retained 占 73.5%，所以多数 class 会让 Accuracy 看起来容易偏高。Dummy model 全猜 retained 已有 73.5% Accuracy，却找不到任何 churner。 | **“Random Forest classifies 75.8% of test customers correctly, but this is only 2.3 percentage points above the all-retained baseline. Therefore, Accuracy alone is not enough for our churn problem.”** |
| **Precision 53.2%** | 系统标记 513 位 churn risks，其中 273 位真的 churn、240 位最终 retained：`273 ÷ 513`。 | 如果 retention team 联络这 513 位客户，约 53% 是真正 churners，约 47% 是 false alerts。Precision 越高，有限的电话、折扣和人工资源浪费越少。 | **“Among customers flagged for retention action, 53.2% actually churned. This controls campaign workload, although 240 retained customers would still receive unnecessary alerts.”** |
| **Recall 73.0%** | 实际有 374 位 churners，模型找到 273 位，漏掉 101 位：`273 ÷ 374`。 | Recall 直接代表系统保护了多少潜在流失客户。73% 表示大部分被发现，但仍有 27% 的 churners 没进入 contact list。若漏掉客户代价最高，应优先提高 Recall。 | **“The model catches 273 of 374 actual churners, giving 73.0% Recall, but it still misses 101 customers who may leave without intervention.”** |
| **F1 61.6%** | 将 Precision 53.2% 与 Recall 73.0% 合成一个 balance score；其中一个很低时，F1 也会被拉低。 | Telco retention 同时不希望漏掉太多 churners，也不希望 contact list 塞满 false alerts，所以我们用 F1 作为主要 model-selection metric。RF 的 F1 是四个模型最高，但只领先约 0.7–0.9 pp，优势存在但不巨大。 | **“Random Forest has the highest F1 at 61.6%, meaning it gives our best tested balance between catching churners and limiting false alerts, although the four models are close.”** |
| **AUC 0.839** | 使用所有 possible thresholds 检查 ranking，不直接使用某一个 TN/FP/FN/TP matrix。 | 它衡量 model 是否通常把真正 churners 排在 retained customers 前面，适合先建立 risk-priority list，再由公司决定 contact threshold。0.839 不是 83.9% Accuracy，也不是客户会 churn 的 probability。 | **“An AUC of 0.839 shows that Random Forest separates higher-risk churners from retained customers well across thresholds, which supports customer prioritisation rather than one fixed contact rule.”** |

**五个 metrics 合起来的完整结论 / Combined metric summary**

> Random Forest is not selected simply because its Accuracy is 75.8%. It catches 73.0% of actual churners, while 53.2% of its alerts are correct. Its F1 of 61.6% is the best balance among our four models, and its AUC of 0.839 gives the strongest overall risk ranking. However, it still misses 101 churners and creates 240 false alerts, so the final threshold must reflect the company's retention capacity and the cost of missed customers.

**Model Ranking bar graph 怎样看 / How to read the ranking graph**

| Screen element | Reading rule |
|---|---|
| Accuracy / Precision / Recall / F1 / AUC buttons | 选择用哪一把 metric ruler 排名，不是在选择 model。 |
| One horizontal bar per model | 每条 bar 是一个 active model 的 test score。 |
| Longer bar | 在当前 selected metric 上表现较高。 |
| Star and dark outline | 标记顶部选择的 active model，例如 RF；不代表它必定是第一名。 |
| Leader and “#1 of 4” cards | 左边显示当前 metric 第一名，右边显示 active model 的位置。 |

选择 F1 时 RF 61.6% 排第一；切到 Recall 后 DT 77.0% 排第一。Graph 改变不是模型突然变好或变坏，而是我们改用了另一把评价尺。

When the selected metric changes, the bars are reordered because the business question has changed—not because any model was retrained.

| Model | Accuracy | Precision | Recall | F1 | AUC | 我们怎样理解 / Project interpretation |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 75.2% | 52.4% | 71.9% | 60.7% | 0.833 | 简单 baseline，表现接近复杂模型。 |
| Decision Tree | 73.7% | 50.3% | **77.0%** | 60.9% | 0.823 | Recall 第一，找出最多 churners，但 false alerts 最多。 |
| Random Forest | **75.8%** | **53.2%** | 73.0% | **61.6%** | **0.839** | Precision、F1、AUC 第一，作为 default。 |
| XGBoost | 75.3% | 52.5% | 72.2% | 60.8% | 0.830 | 强 benchmark，但没有超过 RF。 |

Precision 高适合 retention team capacity 有限；Recall 高适合漏掉 churner 的成本特别高；F1 平衡两边，所以是我们的主要 selection metric。

当老师在 bar graph 上要求 “summarise this metric” 时，应比较四个 models 并说明业务后果：

| Selected graph | 应该怎样 summary / What to say |
|---|---|
| **Accuracy** | RF 75.8% 第一，但只比 73.5% dummy baseline 高 2.3 pp；因此整体正确率必须配合 churn-specific metrics。 / RF leads, but Accuracy alone is weak evidence under class imbalance. |
| **Precision** | RF 53.2% 第一，表示它的 contact list 比其他 models 稍微集中于真正 churners；但仍有 240 false alerts，campaign efficiency 仍有限。 |
| **Recall** | DT 77.0% 第一，比 RF 多找 15 位 churners；代价是多 44 个 false alerts。因此 DT 适合最怕漏客户的情境，不一定适合 capacity-limited team。 |
| **F1** | RF 61.6% 第一，但四个 models 只相差约 0.9 pp；RF 是 best tested balance，不是压倒性胜出。 |
| **AUC** | RF 0.839 第一，四个 models 都在 0.823–0.839，说明它们的 ranking ability 都不错，RF 只有小幅领先。 |

This is the difference between **presenting a number** and **summarising a metric**: state the leader, compare the gap, translate it into customers or workload, and mention the trade-off.

**ROC & PR Curves 怎样看 / How to read the two curve graphs**

| Graph | Axis and good direction | 我们的结果 / Our result |
|---|---|---|
| **ROC curve** | X = False Positive Rate，Y = Recall；curve 越靠左上越好，灰色 diagonal 是 chance。 / Closer to top-left is better. | RF green line 的 AUC 0.839 最高，表示整体 discrimination 最好。 |
| **Precision–Recall curve** | X = Recall，Y = Precision；越靠右上越好。灰色 0.265 是 dataset churn prevalence，也是 no-skill reference。 | Average Precision：LR 0.650、RF 0.648、XGB 0.644、DT 0.600。 |

LR 的 AP 只比 RF 高 0.002，但 RF 同时领先 F1、Precision、ROC-AUC 和整体 error balance，所以我们没有只根据一张 PR graph 更换 default model。

| Model | TN | FP | FN | TP | 与 Random Forest 的差别 / Difference from RF |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 791 | 244 | 105 | 269 | 多 4 个 FP，多漏 4 位 churners。 |
| Decision Tree | 751 | 284 | **86** | **288** | 多找出 15 位 churners，但多 44 个 false alerts。 |
| Random Forest | **795** | **240** | 101 | 273 | False alerts 最少，F1 balance 最好。 |
| XGBoost | 791 | 244 | 104 | 270 | Error pattern 与 LR 很接近。 |

若漏掉客户的成本远高于多打一通电话，可以选 Decision Tree；若 retention team 资源有限，Random Forest 更适合作为 default prioritisation model。没有一个模型在所有 business situations 都绝对最好。

There is no universally best model. The preferred model depends on the cost of missed churners versus unnecessary contacts.

### 3.2 Explainability 怎样联系 Telco signals / Relating explanations to telco signals

Explainability tab 回答：**模型在判断 Telco customer risk 时，整体最依赖哪些客户资料？** 它不是另一种 performance test，也不会告诉我们某个 feature 一定导致 churn。

The tab shows which Telco signals the fitted models rely on globally; it does not test Accuracy or prove causation.

**为什么特别比较 Random Forest 和 XGBoost？ / Why compare these two?**

1. Random Forest 是我们的最终推荐模型；XGBoost 是表现非常接近的另一种 advanced ensemble candidate。比较它们可以检查：RF 的结论是否只属于自己，还是另一种复杂模型也看到相似的 churn signals。
2. 两者虽然都使用 many trees，但学习方式不同。如果它们仍把 contract、charge/tenure、internet 和 payment 放在前面，说明这些 signals 在不同 modelling approaches 下重复出现。
3. 它们都能用同一种 horizontal importance bar 来表达 global reliance。Logistic Regression 要看有正负方向的 coefficients，Decision Tree 更适合看具体 rules；强行放进同一张 bar graph 会混合不同意思的数值。

Random Forest is the selected model and XGBoost is the closest alternative tree ensemble. Comparing them checks whether important Telco signals remain similar under a different modelling approach. LR and DT are not ignored; they require different explanation formats in Active Model Explanation.

老师问时可以简单回答：

> We compare Random Forest with XGBoost because Random Forest is our recommended model and XGBoost is another strong ensemble candidate that learns patterns differently. If both models highlight similar Telco features, the drivers are less likely to be unique to one algorithm. Logistic Regression and Decision Tree are still available, but they are explained using coefficients and decision rules instead of the same importance graph.

**Ensemble Comparison graph 从上到下怎样看**

| 画面 / Display | 它告诉我们什么关于 Telco churn / What it tells us |
|---|---|
| RF leader: ChargesToTenureRatio 0.175 | RF 最依赖“monthly charge 相对于 tenure”的信息。它特别关注费用与客户资历是否不匹配，例如资历很短但费用很高的 profile。 |
| XGB leader: ContractRiskScore 0.262 | XGB 最依赖 contract commitment；month-to-month、one-year、two-year 的差别是它首先关注的 churn signal。 |
| Green vs purple bars | 同一行比较两个 models 对同一 feature 的 reliance。Bar 较长只代表该 model 更依赖它，不代表 probability 增加同样数值。 |
| 8 of 10 shared drivers | 两个 top-10 中有八个相同 features，说明 contract、charges、internet、payment 和 support signals 不是只在一个 model 出现。 |

当前 graph 的主题结论是：

- RF 把注意力较平均地分散在 ChargesToTenureRatio、ContractRiskScore、tenure、MonthlyCharges 和 TotalCharges，说明它综合 customer lifecycle 与 charges。
- XGB 更集中在 ContractRiskScore、Fiber optic、Electronic check 和 InternetService_No，说明它更依赖 contract、internet type 与 payment pattern。
- 两者都重视多个相同 Telco signals，但优先顺序不同。这就是 graph 真正要展示的 model difference。

The comparison shows shared Telco drivers but different priorities: RF is broader across lifecycle and charges, while XGB is more concentrated on contract and service type.

**Active Model Explanation graph 怎样看**

选择 RF 时，每条绿色 bar 是一个 feature；越长表示 RF 在所有 customers 上越依赖它。`ChargesToTenureRatio = 0.175` 是全局第一，不代表它令某一位客户的 churn probability 增加 17.5%。切换 active model 后：

| Active model | 页面为什么用这种解释 / Why this explanation fits |
|---|---|
| Logistic Regression | 显示 coefficients 和 odds ratios，因为它的重点是每个 feature 的 fitted direction；但高 VIF 令独立 coefficient interpretation 不稳定。 |
| Decision Tree | 显示 importance，并可展开 top decision rules，因为单棵树可以沿着判断路径阅读。 |
| Random Forest | 显示 many trees 合起来的 global importance，没有一条简单的单一 rule path。 |
| XGBoost | 显示逐步建立的 trees 最后形成的 global importance。 |

无论哪一种解释，都只说明 model reliance，不是当前 93% prediction 的 local breakdown，也不是因果关系。

### 3.3 Reliability 与 5-fold validation / Reliability and five-fold validation

Reliability tab 回答：**我们现在看到的 model performance，是稳定的 Telco pattern，还是刚好遇到一组比较容易的 customers？**

**Train–Test Context graph 告诉什么**

每一行是一个 model，圆点是它对 training customers 的 F1，diamond 是它对 1,409 unseen test customers 的 F1；中间 line 越长，代表从熟悉数据到新客户下降越多。

Random Forest 从 train F1 0.883 降到 test F1 0.616，gap 0.268，是最长的 line。这告诉我们 RF 在训练资料上学得很强，但面对真实比例的新客户时表现明显下降，所以必须警惕 generalisation，而不能只展示 training score。

不过 train F1 使用 SMOTE-balanced data，test F1 使用 natural churn proportion，两边不是完全相同条件。因此不能简单说“0.268 全部都是 overfitting”。最终仍以 unseen test F1 和 cross-validation 为主要证据；RF 虽然 gap 最大，test F1 仍是四个模型最高。

The graph warns that RF's training performance does not fully carry over to unseen customers, but the gap is not a pure overfitting penalty because the train and test class distributions differ.

**5-fold graph 为什么对我们的 Telco system 有用**

我们只有一份 historical customer dataset。如果只 split 一次，可能刚好把较容易或较困难的 customers 放进 validation。5-fold 的做法是把 5,634 training customers 分成五组，轮流让不同一组做 validation。它要回答：**换一组 Telco customers 后，model ranking 和 score 是否仍然相近？**

Five-fold validation checks whether the conclusion remains similar when different subsets of Telco customers are used for validation.

| 画面上的东西 / Display | 简单意思 / Plain meaning | 对本 project 告诉什么 / What it tells us |
|---|---|---|
| Five small circles | 同一个 model 面对五组不同 validation customers 的五个 scores。 | 点很分散表示 performance 较依赖抽到哪批 customers；点集中表示较稳定。 |
| Diamond / Active CV mean | 五次 scores 的平均。 | 比单一 split 更能代表 model 在不同 customer samples 上的典型表现。 |
| Horizontal ±1 SD line | 五次结果通常相差多大；越短表示 variation 越少。 | 反映 deployment 到不同 customer batches 时，performance 可能有多稳定。 |
| Mean metric leader | 当前 selected metric 的最高平均 model。 | 检查 RF 的 test-set优势是否也在 repeated validation 中出现。 |

在 F1 view 中：

| Model | Five-fold result | 对 Telco system 的解释 / Interpretation |
|---|---:|---|
| Random Forest | **62.8% ± 2.8 pp** | 平均 F1 最高，支持它作为 default；但 variation 最大，说明不同 customer groups 会影响它的 balance。 |
| XGBoost | 62.3% ± 1.6 pp | 平均只比 RF 低 0.5 pp，而且更稳定，是很接近的 alternative。 |
| Logistic Regression | 62.2% ± 2.0 pp | 简单 baseline 仍接近 ensembles，说明复杂模型的 improvement 不大。 |
| Decision Tree | 61.1% ± 1.6 pp | Variation 较小，但平均 F1 最低；稳定不等于最好。 |

正确 summary 不是只说 “RF CV F1 is 62.8%”，而是：

> Random Forest has the highest five-fold mean F1 at 62.8%, supporting its overall selection across different Telco customer subsets. However, its 2.8 percentage-point standard deviation is the largest, so its performance varies more between customer groups than XGBoost or Decision Tree.

Accuracy view 同样用相同方式读：RF 77.4% ± 1.4、XGB 77.0% ± 0.8、LR 76.6% ± 0.8、DT 74.0% ± 1.7。其他 Precision、Recall 和 AUC buttons 不是新的 tests，只是用另一把 metric ruler 重新计算同样五组 predictions。

**为什么使用 5，而不是 10、2 或 0？**

- 5-fold：每次用约 80% training customers 学习、20% validation，能够重复检查五组客户，同时四个 models 和 fold-level SMOTE 的运行时间合理。
- 10-fold：可以使用更多 data training，但需要大约两倍 model fits；不是错误，只是本 project 没有必要增加这部分计算。
- 2-fold：每次只用一半 customers training，而且只有两个 validation scores，结论更容易受 split 影响。
- 0-fold：没有 validation group，无法进行 cross-validation。

每个 fold 都尽量保持相近 churn proportion；SMOTE 只在该 fold 的 training customers 内执行。CV mean 与 held-out test score 不同是正常的，因为 CV 在 training portion 内轮流验证，held-out test 是最后一次独立 evaluation。

### 3.4 Threshold 怎样改变 retention operation / How threshold changes operations

Threshold tab 回答：**模型已经给出 probability 后，公司要从哪里开始把客户放进 contact list？** 模型输出 probability，threshold 决定最终 class：

```text
probability ≥ threshold  → predicted churn
probability < threshold  → predicted retained
```

移动 slider 不会 retrain model，也不会改变原始 probability；它只改变哪些客户会被放进 retention contact list。

Changing the threshold changes the contact decision, not the fitted model.

Threshold 降低时，更多客户被标记，通常 Recall 上升、FN 减少，但 FP 增加、Precision 下降；threshold 提高时则相反。因此 threshold 应根据公司对 missed churner 与 unnecessary contact 的实际成本决定。

Lower thresholds catch more churners but usually create more false alerts. Higher thresholds are more selective but miss more churners.

Slider 下方三张 cards 会即时显示当前 threshold 的 Precision、Recall 和 F1；下一排显示 FN、FP 和 Customers flagged。绿色或红色 delta 是相对于默认 0.50 的变化。例如 threshold 仍是 0.50 时，所有 delta 都是 0。

The cards translate the selected threshold into model quality and actual retention workload.

**Confusion Matrix graph 怎样看 / How to read the matrix**

- Rows 是实际结果：上面 Actual Retained，下面 Actual Churn。
- Columns 是模型决定：左边 Predicted Retained，右边 Predicted Churn。
- Blue TN 与 green TP 是正确决定；amber FP 与 red FN 是两种错误。
- `Customers` 显示人数；`Rate within actual class` 把每一 actual row 转成百分比。

| Outcome | Count | 在 Telco operation 中的意思 / Operational meaning |
|---|---:|---|
| TN | 795 | 正确没有把 retained customer 放入 contact list。 |
| FP | 240 | 对最终 retained 的客户发出 unnecessary alert。 |
| FN | 101 | 漏掉实际 churner。 |
| TP | 273 | 成功找出实际 churner。 |

因此 actual churners = `101 + 273 = 374`，Recall = `273 / 374 = 73.0%`；predicted churn = `240 + 273 = 513`，Precision = `273 / 513 = 53.2%`。

The confusion matrix connects metrics to real customer counts, making the retention workload and missed opportunities visible.

`Prediction Flow` 不是另一份结果。左边是实际 1,035 retained 与 374 churn customers，右边是模型预测的 896 retained 与 513 churn；带子的宽度代表流向该结果的人数。它把相同 TN、FP、FN、TP 画成 Sankey view。Customers 显示人数；Share of test set 则除以 1,409，适合看整体比例。

Prediction Flow is the same confusion information visualised as customer movement.

Threshold curve 的 x-axis 是 threshold，y-axis 是 metric score。蓝线 Precision 通常随 threshold 上升，黄线 Recall 通常下降，绿线 F1 显示两者平衡；vertical line 标出 slider 当前位置。当前 test set 的 exploratory F1-optimal threshold 约为 0.54，但不能直接设为 production threshold，因为它是查看 test outcomes 后选出的；还需要 separate validation 和真实 contact/churn cost analysis。

The 0.54 threshold is exploratory and should not be treated as production-ready without separate validation.

3D threshold graph 只是把相同 threshold、Precision、Recall 和 F1 path 放进 3D view，不是额外证据。短 presentation 应优先展示 confusion matrix 和 2D threshold curve。

The 3D view adds no new evaluation result and is optional during a short presentation.

### 3.5 Models 页面现场说法 / Models-page presentation script

建议控制在两分钟：

1. 在 Performance 选择 F1：`Random Forest has the best F1 balance, while Decision Tree has the highest Recall.`
2. 切换 Recall：`The leader changes because Recall rewards catching churners, even when false alerts increase.`
3. 打开 ROC & PR：`ROC measures overall ranking; PR focuses on the minority churn class.`
4. 打开 Ensemble Comparison：`Random Forest spreads reliance across charge, lifecycle and contract signals, while XGBoost concentrates more on contract and internet type.`
5. 打开 5-Fold F1：`The circles are fold scores, the diamond is the mean and the line is ±1 SD. We chose five folds as a stability–computation trade-off.`
6. 打开 threshold 0.50 confusion matrix：`Random Forest catches 273 churners, misses 101 and produces 240 false alerts.`
7. 用业务结论结束：`A lower threshold catches more churners but increases the contact workload, so the final threshold depends on retention cost and capacity.`

| Question | Short answer |
|---|---|
| Which metric is most important? | F1 is our main selection metric because it balances Precision and Recall on an imbalanced churn problem. We also inspect Recall, AUC and operational error counts. |
| Why not select Decision Tree? | It catches 15 more churners than RF but creates 44 more false alerts. RF provides the better default balance. |
| Why compare only Random Forest and XGBoost in Ensemble Comparison? | RF is our recommended model and XGB is a close alternative ensemble that learns differently. Comparing them checks whether important Telco signals are shared across two advanced approaches. LR coefficients and DT rules are shown separately because their explanation values mean something different. |
| Why K = 5? | It gives five validation estimates while training on 80% of the training data each round, with reasonable computation for four models and fold-level SMOTE. |
| Does low SD mean best model? | No. SD measures variation; mean performance and variation must be considered together. |
| Why is CV F1 different from test F1? | CV averages five validations inside training data; test F1 is one final result on a separate held-out set. |
| Does the train–test gap prove overfitting? | It is a warning, but the train and test F1 values use different class distributions, so it is not a pure overfitting score. |
| What does changing threshold do? | It changes probability-to-class decisions and contact workload; it does not retrain the model. |
| Why not use threshold 0.54 immediately? | It was found by exploring this test set and needs separate validation plus business-cost analysis. |
| Does importance show causation? | No. It shows global model reliance in historical data. |

最后每位组员必须记住：

1. 93% 是单一客户的 estimated churn probability，不是 Accuracy，也不是 certainty。
2. Random Forest 是 default，因为它在我们的 test data 上有最好的 F1、AUC 和 error balance。
3. Decision Tree Recall 最高，但会制造更多 false alerts。
4. Feature importance 是 global reliance，不是 local contribution 或 causation。
5. K-fold 中 circles 是 fold scores、diamond 是 mean、横线是 ±1 SD。
6. Threshold 越低，通常抓到更多 churners，但 contact workload 越大。
7. Streamlit 的 selectors 与 sliders 不会重新训练模型。
