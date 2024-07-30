from openai import OpenAI
import random
import torch
import pickle

client = OpenAI(
  api_key='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'
)

system_msg = "You are an expert at identifying room types based on the object detected. Give short single responses."
question = "\n What type of room is most likely? Choose among basement, bathroom, bedroom, living room, home lobby, office, lab, kitchen, dining room."
queries = {}
with open('scene/llm_single.pkl', 'rb') as f: 
  queries = pickle.load(f)

def classify_compose(objects):
  return

def classify_cot(objects):
  better_objects = classify_compose(objects)
  answer = classify_llm(better_objects)
  return answer

def classify_llm(objects):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  answer = call_llm(objects)
  counts = torch.zeros(9)
  for a in answer:
    s = parse_response(a)
    counts[random_scene.index(s)] += 1
  return random_scene[counts.argmax()]

def call_llm(objects):
  r = []
  for o in objects:
    if o == 'skip' or o == 'ball': 
      continue
    prompt = f"There is a {o}."
    if o in queries.keys():
      r.append(queries[o])
      continue
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                  {"role": "system", "content": system_msg},
                  {"role": "user", "content": prompt + question}
                ],
                top_p=1e-8
              )
    if response.choices[0].finish_reason == 'stop':
      ans = response.choices[0].message.content.lower()
      print(ans)
      queries[o] = ans
      with open('scene/llm_single.pkl', 'wb') as f:
        pickle.dump(queries, f)
      r.append(ans)
    else: 
      raise Exception("LLM failed to provide an answer") 
  return r

def classify_llm_all(objects):
  answer = call_llm(objects)
  answer = parse_response(answer)
  return answer

def call_llm_all(objects):
  objects.sort()
  objects = list(set(objects))
  if 'skip' in objects: objects.remove('skip')
  if 'ball' in objects: objects.remove('ball')
  user_list = ", ".join(objects)
  if len(objects)==1: prompt = f"There is a {user_list}."
  else: prompt = f"There are {user_list}."
  if user_list in queries.keys():
    return queries[user_list]
  response = client.chat.completions.create(
              model="gpt-4o", # gpt-4o
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt + question}
              ],
              top_p=1e-8
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content.lower()
    print(ans)
    queries[user_list] = ans
    with open('scene/llm.pkl', 'wb') as f:
      pickle.dump(queries, f)
    return ans
  raise Exception("LLM failed to provide an answer") 

def parse_response(answer):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  for s in random_scene:
    if s in answer: return s
  raise Exception("LLM failed to provide an answer") 
