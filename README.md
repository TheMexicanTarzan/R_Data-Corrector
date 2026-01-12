# R_Data-Corrector

To run the data cleaning pipeline:
- Create a new environment
- Run the environment.yml file
- Paste the input files into R_Data-Corrector\Input\Data
- Paste the universe information file to R_Data-Corrector\Input\Universe_Information
- Run the main

To replicate the forensic analysis:
- Select a language model with research capabilities (Gemini 3 Pro deep research mode was used for the report)
- Attach one of the the error logs found in R_Data-Corrector\Output\full_pipeline\error_logs
- Go to R_Data-Corrector\Analysis\prompts, copy the prompt which corresponds to the selected error log and paste it as an instruction

**Note:** As mentioned in the report, the soft filters from validate_financial_equivalencies are numerous and contain a very high number of false positives. Therefore, It is advised to remove the logs from this category so the LLM does not get distracted from genuine logs during analysis.
