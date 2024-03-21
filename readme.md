Fine Tunning LLM with domain data fom the training guides
This is work in progress.Things completed so far :-
    1. Base Code prepared using Mistral AI LLM.
    2. The code now can read from PDF and contatenate all the pdf.s

Next Step:-
    As this is a pdf document it doesn't have any PAD tokens.So in next session will add PAD tokems so that the tokenizer can tokeninize the pdf text

7th Mar, 2024

    As per this commit the encoder is working as well as the training loop.I understand the encoder part well, thanks to the video tutorial by Andre on tokenizer.
    In next session will try to understand the training loop.

18th Mar, 2024
    
    Code does the fine tuning and generates some random text.But there is some issue related to the tokeneizer pad tokens and eos tokens which is causing a text to be repeated multiple times.

21st Mar, 2023

    The code is done and something is getting fine tuned.I havd a MAC and the finetraining for a 16mb pdf is running for last 24 hrs.The code is very suboptimal and I am sure there would be ways to do it faster.But still to bring this to a raesonale level I need more compute.I don't want to use any cloud instance so next stpes would be to buy some gpus to check the effectiveness of the training.

