## ** Model Training Plan** 

## ** Objectives of Model Training** 

The model training component exists to support four application capabilities: 

1. Identify individual ingredients from OCR extracted ingredient text. 

2. Classify ingredients according to nutritional, functional, and source-based categories. 

3. Detect allergens and dietary risk components with high recall. 

- Produce predictions sufficiently lightweight for real-time on-device execution. 

4. 

The training process therefore prioritizes: 

- safety over aggressiveness, 

- recall over precision for allergens, 

- explainability over model complexity, 

- mobile deployment feasibility over state-of-the-art benchmark performance. 

The models are intended to assist users rather than replace professional dietary advice. 

## ** Deliverables** 

At completion, the model training phase shall produce: 

## **Deliverable 1** 

A semantic ingredient classification model capable of assigning one or more labels to an ingredient. 

Example: 

```
MSG
→ Flavor Enhancer
→ Sodium Compound
→ Food Additive
```

```
Caseinate
→ Milk Derivative
→ Animal Derived
→ Protein Source
```

1 

## **Deliverable 2** 

An allergen detection model capable of identifying: 

- explicit allergens, 

- implicit allergens, 

- allergen derivatives, 

- contextual allergen references. 

Example: 

```
Contains milk
```

```
Whey Protein Concentrate
```

```
Soy Lecithin
```

## **Deliverable 3** 

A rule engine containing: 

- allergen synonyms, 

- ingredient aliases, 

- Filipino ingredient variants, 

- regulatory terminology. 

## **Deliverable 4** 

TensorFlow Lite versions of all models suitable for Android deployment. 

Target model size: 

- Classification model < 40 MB 

- Allergen model < 40 MB 

Combined inference memory target: 

< 120 MB RAM. 

2 

## ** Success Criteria** 

The project will consider model development successful if the following minimum targets are achieved. 

|Task|Metric|Target|
|---|---|---|
|Ingredient Classifcation|Macro F1|≥0.90|
|Ingredient Classifcation|Micro F1|≥0.90|
|Allergen Detection|Recall|≥0.90|
|Allergen Detection|Precision|≥0.75|
|Allergen Detection|F1 Score|≥0.80|
|OCR Pipeline|Ingredient Extraction Accuracy|≥95%|
|Mobile Inference|Average Runtime|< 1 second|



Failure to achieve these targets triggers an additional annotation cycle and retraining iteration. 

## ** Dataset Composition** 

The dataset shall consist of approximately 2,000 unique ingredient entries. 

Target composition: 

|Source|Approximate Count|
|---|---|
|Open Food Facts|1,400|
|Philippine Products|400|
|FDA Verifed Labels|200|



Priority shall be given to products commonly found in Philippine supermarkets. 

Examples include: 

• chips • biscuits • instant noodles • candies • crackers • cereal products • processed snack foods 

3 

Duplicate products shall be removed. 

Multiple package sizes of the same product count as one entry. 

## ** Annotation Objectives** 

Each ingredient token shall contain: 

## **Semantic Labels** 

Examples: 

|Ingredient|Labels|
|---|---|
|Sugar|Added Sugar|
|Sodium Benzoate|Preservative|
|Soy Lecithin|Emulsifer|
|Palm Oil|Fat|
|Gelatin|Animal Derived|



## **Allergen Labels** 

Examples: 

|Ingredient|Allergen|
|---|---|
|Caseinate|Milk|
|Albumin|Egg|
|Wheat Flour|Wheat|
|Soy Lecithin|Soy|



## **Detection Type** 

Each allergen annotation must specify: 

- Explicit 

4 

• Implicit • Derivative 

Example: 

|Ingredient|Allergen|Type|
|---|---|---|
|Milk Powder|Milk|Explicit|
|Whey|Milk|Derivative|
|Soy Lecithin|Soy|Derivative|



## ** Dataset Split** 

The dataset will be split before training. 

|Dataset|Percentage|
|---|---|
|Training|80%|
|Validation|10%|
|Testing|10%|



No product may appear in multiple splits. 

This prevents information leakage. 

## ** Handling Class Imbalance** 

Certain allergens occur infrequently. 

Examples: 

- shellfish, • tree nuts, • fish derivatives. 

To prevent model bias: 

- minority classes shall be oversampled, 

- weighted loss functions shall be applied, 

- augmentation shall prioritize underrepresented labels. 

5 

## ** Data Augmentation** 

The following augmentation methods are permitted: 

## **OCR Noise Simulation** 

Example: 

```
sodium benzoate
```

becomes 

```
sod1um benzoate
```

## **Synonym Replacement** 

Example: 

```
sugar
```

becomes 

```
sucrose
```

## **Formatting Variations** 

Example: 

```
milk powder, sugar, cocoa
```

becomes 

```
milk powder; sugar; cocoa
```

6 

## **Parenthetical Expansion** 

Example: 

```
milk solids (caseinate)
```

becomes 

```
milk solids caseinate
```

No augmentation may alter allergen meaning. 

## **Model Selection Rationale** 

Several candidate architectures were considered. 

|Model|Advantages|Disadvantages|
|---|---|---|
|DistilBERT|Good accuracy|Larger memory|
|TinyBERT|Small model|Lower accuracy|
|MobileBERT|Good balance|Slightly slower|
|BiLSTM|Lightweight|Poor context handling|



MobileBERT is selected because it provides: 

- contextual understanding, 

- acceptable inference speed, 

- TensorFlow Lite compatibility, 

- mobile deployment feasibility. 

- 

## **Training Configuration** 

Initial training configuration: 

7 

|Parameter|Value|
|---|---|
|Optimizer|AdamW|
|Learning Rate|2e-5|
|Epochs|15|
|Batch Size|16|
|Weight Decay|0.01|
|Dropout|0.1|
|Max Sequence Length|128|



Acceptable search ranges: 

|Parameter|Range|
|---|---|
|Learning Rate|1e-5 to 5e-5|
|Epochs|5 to 20|
|Batch Size|8 to 32|
|Weight Decay|0 to 0.1|



## **Training Procedure** 

For each experiment: 

1. Load training data. 

2. Tokenize ingredient sequences. 

3. Generate attention masks. 

4. Train model. 

5. Evaluate validation metrics. 

6. Save checkpoint. 

- Compare against best model. 

7. 

8. Continue until convergence or early stopping. 

Early stopping criteria: 

- validation loss fails to improve for three consecutive epochs. 

8 

## **Checkpoint Policy** 

The following artifacts must be stored: 

- model weights, 

- tokenizer version, 

- training configuration, 

- dataset version, 

- label schema, 

- evaluation metrics. 

This ensures reproducibility. 

## **Error Analysis** 

After evaluation, all incorrect predictions shall be categorized into: 

- OCR failure, 

- synonym failure, 

- parser failure, 

- implicit allergen miss, 

- ambiguous ingredient, 

- unsupported ingredient. 

Error frequencies will determine future development priorities. 

## **Deployment Pipeline** 

Deployment sequence: 

```
PyTorch
↓
ONNX Export
↓
TensorFlow Conversion
↓
TensorFlow Lite Conversion
↓
Quantization
↓
Android Integration
```

9 

If conversion causes unacceptable accuracy degradation: 

Fallback option: 

```
PyTorch Mobile
```

## **Risk Management** 

|Risk|Mitigation|
|---|---|
|Insufcient local data|Increase Philippine data collection|
|Poor OCR quality|Improve preprocessing|
|Model too large|Apply quantization|
|Slow inference|Reduce sequence length|
|Poor allergen recall|Increase rule-based weighting|



## **Final Acceptance Criteria** 

A model release is approved only if: 

- target metrics are achieved, • inference operates offline, 

- memory consumption remains acceptable, 

- inference latency remains below target, 

- false negative allergen rates remain within acceptable limits. 

Only approved releases are converted for production deployment. 

10 

