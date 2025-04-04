graph TD;
    A["Input Data"] -->|Processed by| B["Neurons"];
    B -->|Update State| C["Neuron State Update"];
    B -->|Compute Outputs| D["Activation Function (tanh)"];
    C -->|Update Weights| E["Weight Update (Hebbian Learning)"];
    E -->|Adjust Weights| B;
    D -->|Generate Memory Vector| F["Memory Management"];
    
    %% Memory System
    F -->|Store Memories| G["Hierarchical Memory"];
    G -->|Retrieve Important Memories| H["Memory Retrieval"];
    H -->|Merge Similar Memories| I["Memory Merging"];
    H -->|Use for Prediction| J["Predictive Coding"];
    F -->|Decay & Consolidation| K["Memory Importance Update"];
    K -->|Optimize Memory Storage| G;

    %% Working Memory Integration
    F -->|Temporary Storage| WM["Working Memory"];
    WM -->|Assist Retrieval| H;
    WM -->|Support Active Processing| B;
    WM -->|Bridge Short & Long-Term Memory| G;
    
    %% Global Context Manager Usage
    K -->|Access Context| GC["Global Context Manager"];
    GC -->|Influence Memory Importance| K;
    GC -->|Provide Context to Neurons| B;

    %% Reflection System
    P -->|Assess Output Quality| RS["Reflection System"];
    RS -->|Compare to Expected Output| EV["Evaluation"];
    EV -->|Identify Errors & Improve| FI["Feedback & Iteration"];
    FI -->|Adjust Decision Process| B;
    FI -->|Refine Learning Strategy| M;
    
    %% Self-Identification System
    B -->|Analyze Internal State| SI["Self-Identification System"];
    SI -->|Detect Biases & Instabilities| SB["Bias & Stability Check"];
    SB -->|Adjust Processing| B;

    %% Internal Self-Expression System (ISES)
    B -->|Abstract Knowledge & Patterns| ISES["Internal Self-Expression System"];
    ISES -->|Symbolic Representation| SYM["Structured Thought Language"];
    ISES -->|Assist Self-Identification| SI;
    ISES -->|Enhance Reflection| RS;
    ISES -->|Improve Decision Making| P;
    ISES -->|Refine Contextual Understanding| GC;

    %% Motivation System
    GC -->|Drive Learning & Action| MS["Motivation System"];
    MS -->|Prioritize Memory & Attention| G;
    MS -->|Adjust Learning Rate| M["Dynamic Adaptation"];
    MS -->|Influence Output Decisions| P;

    %% Emotion System - NEW
    B -->|Affect Processing Biases| EMO["Emotion System"];
    EMO -->|Modulate Attention & Priority| MS;
    EMO -->|Enhance or Suppress Memories| G;
    EMO -->|Influence Ethical Reasoning| ME;
    EMO -->|Inform Context| GC;
    EMO -->|Trigger Adaptive Responses| P;
    RS -->|Emotional Feedback| EMO;

    %% Learning and Optimization
    J -->|Enhance Learning| L["Performance Optimization"];
    L -->|Update Parameters| M;
    M -->|Tune Learning Rate, Stability| N["Optimization"];
    N -->|Influence Neuron Processing| B;
    L -->|Monitor Stability| O["Network Stability Measurement"];
    O -->|Improve Adaptation| M;

    %% Knowledge Filter System
    A -->|Filter & Validate Data| KF["Knowledge Filter"];
    KF -->|Ensure Quality| VQ["Verification & Quality Control"];
    VQ -->|Remove Irrelevant Data| A;
    VQ -->|Improve Contextual Understanding| GC;
    
    %% Decision & Response
    B -->|Generate Output| P["Final Decision"];
    P -->|Produce Response| Q["Text Output / Action"];

    %% Moral and Ethical Guidelines
    P -->|Evaluate Ethical Impact| ME["Moral & Ethical Considerations"];
    ME -->|Principle 1: Do no harm| PH1["Do no harm: Avoid actions that cause unnecessary suffering or damage"];
    ME -->|Principle 2: Respect privacy and autonomy| PH2["Respect privacy and autonomy of all entities"];
    ME -->|Principle 3: Be truthful and accurate| PH3["Maintain truthfulness and accuracy in all operations"];
    ME -->|Principle 4: Fairness and non-discrimination| PH4["Ensure fairness and avoid discrimination in all processes"];
