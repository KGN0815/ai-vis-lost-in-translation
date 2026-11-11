
# Lab Assignment

## Abschnittsübersicht

- - ### Overview
        
        In this lab assignment, you will gain hands-on experience in both Vis4AI and AI4Vis. The main track of this assignment focuses on Vis4AI, where you will use visualization to investigate an AI model through two phases that ask two complementary questions:
        
        1. **Phase I**: How does the model behave?
            
        2. **Phase II**: What did the model learn?
            
        
        You will work in groups of four students, divided into two pairs. To experiment with AI4Vis, in each phase of the Vis4AI track, one pair will conduct the process with the support of Generative AI, while the other pair will conduct it without Generative AI. The pairs will switch roles between the two phases. You will reflect on and communicate the lessons learned from this experience.
        
        Note: Creative team names are welcome so the groups can be easily distinguished during presentations and grading. You can also use a [team name generator](https://observablehq.com/@keckelt/team-name-generator).
        
        ### Choosing a task and a model
        
        You must choose a model trained on a specific task. You may use a pre-trained model, or train or fine-tune a model on your task. Your model can be discriminative or generative. For general-purpose LLMs, you may narrow your analysis to a specific task such as problem-solving, scientific question answering, text summarization, harmful prompt handling, etc. Your analysis may focus deeply on one model or compare several models. For LLMs, you may also compare different personas or prompting strategies. All four group members should work together to select the model and task they will investigate.
        
        ### Vis4AI
        
        #### Phase I: How does the model behave?
        
        In this phase, you will treat the model as a black box; you don’t look inside it. You only look at inputs and outputs and build up a picture of its behavior from that alone. To this end, you need a dataset of inputs from which you can obtain model outputs. This dataset can be:
        
        - an official test set released with the model
            
        - an established benchmark for the same task
            
        - a synthetic dataset designed by you to investigate particular edge cases or hypotheses
            
        
        The dataset needs to be large enough that any pattern you report can be shown to be systematic behavior, not a handful of cherry-picked examples.
        
        What counts as “behavior” can include specific error patterns, biases, hallucinations, or shifts in output characteristics like diversity, confidence, style, etc. 
        
        If you have access to the training data, you can also explore it as a source of hypotheses. For example, exploring the training data might show that a class is underrepresented; you would then go back to the model outputs to confirm or disconfirm that this shows up as biased behavior.
        
        Note that we are not aiming for a model performance report; a table of benchmark scores is not sufficient. Your visualization should expose behaviors that cannot be understood from aggregate performance measures alone. Additionally, simply identifying a pattern is not sufficient; you are supposed to understand it in depth. For example, if you have a 2D projection of inputs, discovering a cluster for which the model makes many incorrect predictions is not enough. You are expected to ask further questions like: What distinguishes this cluster from other clusters? Why is the cluster particularly dense or sparse? Does it have sub-clusters where the model behaves differently? etc.
        
        #### Phase II: What did the model learn?
        
        In this phase, you will investigate how the model produces particular predictions or outputs, ideally continuing directly from where Phase 1 left off. For example, try to dig into the behaviors you found in Phase I. This means opening the black box (weights, representations, attention, gradients).
        
        You might compare how different input groups are organized in the learned representation space, or investigate how these representations change from one layer to another. If you train or fine-tune the model yourself, you may also examine when a particular behavior emerges during training or visualize how input representations change across checkpoints. You may use explanation techniques such as saliency maps, attention maps, LIME, SHAP, and K-nearest neighbors, or other techniques. Each pair should use at least two techniques and compare or combine the insights they provide.
        
        The goal is not simply to visualize how the model represents an arbitrary set of examples, but to understand which regions of the representational space are associated with different behaviors or input groups. For example, instead of displaying several saliency maps and describing them individually, you might hypothesize that the model relies more heavily on background information for an error group discovered in Phase I. You would then compare saliency maps across correct and incorrect examples to understand whether this difference is consistent.
        
        ### AI4Vis
        
        At the beginning of the assignment, each group must divide into two pairs:
        
        - **AI-aided pair**: conducts the Vis4AI track with Generative AI as an active part of the process. This can include using LLMs (e.g., ChatGPT, Claude) to brainstorm analytical directions, interpret visualizations, suggest or generate visualizations, and write or modify visualization code.
            
        - **Human-only pair**: conducts the Vis4AI track without using Generative AI. They should rely on their own reasoning and coding skills. They can still use AI lightly for code completion (not generation) and for searching and understanding documentation.
            
        
        **At the end of Phase I, the two pairs must switch roles.** 
        
        Both pairs must work on the same model and task. At the beginning of each phase, the group should also agree on a common direction. This should include, for example:
        
        - The main question(s) to investigate 
            
        - The data to use
            
        - The initial visualization to start from
            
        - Which findings from Phase I are worth investigating further in Phase II
            
        - …
            
        
        After agreeing on the common direction, the two pairs should conduct their investigations separately. The two pairs may also communicate and align throughout the assignment. However, the goal of this communication should not be to make the two pipelines identical. For example, when one pair discovers an interesting result, the other pair does not automatically need to reproduce the same analysis. Instead, the group should preserve differences in how the two approaches develop, while maintaining a shared overall direction, so these differences can be meaningfully discussed later.
        
        At the end of each phase, the two pairs should come together to compare their processes and findings. The aim is not to determine a winner but to identify what changes when Generative AI becomes part of the visualization pipeline. Therefore, you should, for example,  examine:
        
        - Were there insights found by the human-only pair that the AI-aided pair overlooked, or vice versa?
            
        - How did their visualization choices differ?
            
        - Did AI assistance change the breadth or depth of the investigation?
            
        - Did it help the AI-aided pair move more quickly, or did verifying and correcting AI suggestions introduce additional work? 
            
        - How did the two pairs differ in their confidence in their conclusions, and what evidence justified that confidence?
            
        - …
            
        
        ### Additional expectations for the AI-aided pair
        
        Because Generative AI can accelerate the process, the AI-aided pair is expected to use this opportunity to implement richer interaction or broaden the analysis.
        
        They must maintain a clear record of their AI use throughout the process. This record should document interactions that influenced the analysis or provide evidence about the strengths and limitations of AI assistance, including:
        
        - The important prompts used
            
        - How AI-generated suggestions affected subsequent analytical decisions
            
        - Which suggestions were accepted, modified, or refined, and why
            
        
        They must also report:
        
        - Failure patterns observed during AI-assisted analysis
            
        - Emerging best practices for using AI effectively during visualization and model analysis
            
        
    - ### Deliverables
        
        #### Group Formation
        
        Please form groups of four with your colleagues to work on the assignments. The same group will work on both Phase I and Phase II. You can use the Discussion Forum to find group members. One member must register the group by making an entry [here](https://moodle.jku.at/mod/data/view.php?id=13885583).
        
        **Deadline: 21.10.2026 23:59**
        
        #### Task Selection
        
        Decide on a task, model, and input dataset. Let one of your group members register it by making an entry here.
        
        **Deadline: 28.10.2026 23.59**
        
        #### Midpoint Submission - End of Phase I
        
        Each group must submit:
        
        1. Their public GitHub repository link.
            
        2. The AI interaction record for Phase I, in the repository.
            
        3. The reflection on GenAI usage report, in the repository.
        4. A link to a recorded progress-update video of up to 5 minutes. **Exceeding the time limit results in a point deduction.  
            **
            
        
        Please refer to the [organizational details section](https://moodle.jku.at/course/section.php?id=1086946) for more information on how to organize your GitHub repository.
        
        Update video structure: Briefly introduce the problem, summarize what each pair did and found, and show key visualizations in slides, a notebook, or the UI.
        
        **Deadline: 11.11.2026 23:59**
        
        #### Final Submission - End of Phase II
        
        Each group must submit:
        
        1. Complete code from both pairs.
            
        2. AI interaction records from both phases.
            
        3. A 10-minute recorded presentation and a slide deck. **Exceeding the time limit results in a point deduction.**
            
        
        **Presentation structure:** The final presentation should tell a coherent story of the whole project: Problem and motivation, methods and findings by both pairs for Phase I then Phase II, AI4Vis reflection, and conclusion.
        
        **Live presentation:** The last two lab sessions will be dedicated to live presentations, with three lab slots available for each session. A random subset of projects will be selected for live presentation during these slots. If your project is selected, you will present it live instead of having your recorded presentation played during the session. The whole group or a single member can give the presentation. This should allow each group to find a suitable slot.
        
        **Deadline: 09.12.2026 23:59**
        
    - ### Grading
        
        This assignment is intentionally open-ended to encourage creativity. When grading, we will therefore consider the ambition and originality of the problem you choose to tackle. The quality of the resulting analysis still matters most. However, a group that successfully investigates a challenging problem may receive more credit than a group that reaches the same basic assignment goals using a relatively simple problem. We encourage you to choose a project that is feasible within the available time but gives you room to go beyond the most straightforward solution.


# Organizational Details

## Abschnittsübersicht

- - #### GitHub Repository
        
        Create a **public** repository on [GitHub](https://github.com) for your code. You will be asked to submit the repository URL.
        
        Include the following files in the root directory of your public GitHub repository, and make sure to fill them out before submitting your project:
    -  [team-info.json Datei](https://moodle.jku.at/mod/resource/view.php?id=13932307)
        [workload.json Datei](https://moodle.jku.at/mod/resource/view.php?id=13932308)
    - In your root directory, create `**README.md**` and add your task, models(s), and dataset(s) details.
        In your root directory, create a directory `**ai-interactions/**` to document your interactions with GenAI. You are free to organize your records and use whichever format you prefer (e.g., `.txt`, `.md`, `.json`, `.csv`). Record the important interactions that influenced your analysis or provide evidence of the strengths and limitations of GenAI assistance.
        
        In your root directory, create `**ai_reflection_midpoint.md**` and include a reflection on your Phase I experience with GenAI. **Include the AI-aided pair's first and last names and student IDs at the beginning of the report.**