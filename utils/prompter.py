# Prompt Formats are from LongBench. (https://arxiv.org/pdf/2308.14508)
import os
import json
import re
prompt_format = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    # "hotpotqa": "Question: {input}\nAnswer:",
    # "hotpotqa": "<s> Based on the following passages, answer the question.\n{context}\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

specialized_prompt_format = {
    "hotpotqa": "<s> Based on the following passages, answer the question.\n\n<passages>\n{context}\n\n</passages>\n\nQuestion: {input}\nAnswer: ",
}


description = {
    "input": "任务的输入/指令，通常较短，比如QA中的问题、Few-shot任务中的提问等",
    "context": "任务所需的长语境文本，比如文档、跨文件代码、Few-shot任务中的few-shot样本",
    "answers": "由所有标准答案组成的列表",
    "length": "前三项文本的总长度（中、英文分别用字、词数统计）",
    "dataset": "本条数据所属数据集名称",
    "language": "本条数据的语言",
    "all_classes": "分类任务中的所有类别，非分类任务则为null",
    "_id": "每条数据的随机id"
}
        
class LongBenchPromptBuilder:
    def __init__(self, task_name):
        self.task_name = task_name

    def post_process(response, model_name):
        if "xgen" in model_name:
            response = response.strip().replace("Assistant:", "")
        elif "internlm" in model_name:
            response = response.split("<eoa>")[0]
        return response
    
    
    def context_fit_train(self, context, input):
        pattern = re.compile(r"Passage \d+:\n")
        splited_passages = re.split(pattern, context)
        template = "<passage>\n#{title}\n{content}\n</passage>"
        formated_passages = []
        for i in range(1, len(splited_passages)):
            title = splited_passages[i].split("\n")[0]
            content = splited_passages[i].split("\n", 1)[1].strip()
            formated_passages.append(template.format(title=title, content=content))
            # print(template.format(title=title, content=content))
        return "\n\n".join(formated_passages), input
    
    def build_prompt(self, data, fit_train = False):
        if self.task_name not in prompt_format:
            raise ValueError(f"Task name {self.task_name} is not supported.")
        results = []
        for i, item in enumerate(data):
            if fit_train:
                template = prompt_format[self.task_name] if self.task_name not in specialized_prompt_format else specialized_prompt_format[self.task_name]
                context, input = self.context_fit_train(item['context'], item['input'])
            else:
                template = prompt_format[self.task_name]
                context, input = item['context'], item['input']
            results.append(template.format(context=context, input=input))
            # results.append(template.format(input=input)) # NOTE: directly_ask
            # import pdb; pdb.set_trace()
        return results


class ZeroScrollsPromptBuilder:
    def __init__(self, task_name):
        self.task_name = task_name
    
    def build_prompt(self, data):
        results = []
        for item in data:
            input_str = item['input']
            if "Question:\n" in input_str:
                input_str = input_str.strip().replace("Question:\n", "Question: ")
            elif "Question and Possible Answers:\n" in input_str:
                input_str = input_str.strip().replace("Question and Possible Answers:\n", "Question and Possible Answers: ")

            results.append(input_str)
        return results

class DetectorPromptBuilder:
    def build_prompt(self, data):
        results = []
        for item in data:
            results.append(item['Prompt'])
        return results