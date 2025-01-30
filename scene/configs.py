from openai import OpenAI
import random
import torch
import pickle
import dataset

from collections import Counter

objects = dataset.objects

scene_objects = {
    'lobby': dataset.lobby_objects,
    'lab': dataset.lab_objects,
    'bathroom': dataset.bathroom_objects,
    'bedroom': dataset.bedroom_objects,
    'living': dataset.living_objects,
    'kitchen': dataset.kitchen_objects,
    'dining': dataset.dining_objects,
    'office': dataset.office_objects,
    'basement': dataset.basement_objects
}

def find_most_likely_scene(objects):
    # Create a counter for each scene
    scene_counter = Counter()

    # Count occurrences of objects in each scene
    for obj in objects:
        for scene, scene_objs in scene_objects.items():
            if obj in scene_objs:
                scene_counter[scene] += 1

    # Find the scene with the highest count
    most_likely_scene = scene_counter.most_common(1)
    return most_likely_scene[0][0] if most_likely_scene else "bedroom"

client = OpenAI(
  api_key='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'
)

system_msg = "You are an expert at identifying room types based on the object detected. Give short single responses."
question = "\n What type of room is most likely? Choose among basement, bathroom, bedroom, living room, home lobby, office, lab, kitchen, dining room."
queries = {}
with open('/Users/alaiasolko-breslin/Penn/research/blackbox-learning/scene/scene/llm.pkl', 'rb') as f: 
  queries = pickle.load(f)

def classify_compose(objects):
  objects = list(set(objects))
  scene = find_most_likely_scene(objects)
  new_objects = scene_objects[scene][:5]
  full_objects = objects + new_objects
  return list(set(full_objects))
  # return [scene]

def classify_cot(objects):
  better_objects = classify_compose(objects)
  answer = classify_llm(better_objects)
  return answer

def classify_llm(objects):
  objects = list(set(objects))
  scene = find_most_likely_scene(objects)
  return scene
  # if 'wardrobe' in objects or 'bed' in objects: return "bedroom"
  # elif 'machines' in objects or 'printer' in objects: return 'office'
  # elif 'sofa' in objects or 'tv' in objects: return 'living'
  # elif 'toilet' in objects or 'sink' in objects: return 'bathroom'
  # elif 'storage objects' in objects or 'washer' in objects: return 'basement'
  # elif 'sink' in objects or 'refrigerator' in objects or 'oven' in objects or 'stove' in objects: return 'kitchen'
  # return "kitchen"
  # answer = call_llm(objects)
  # answer = parse_response(answer)
  # return answer

def call_llm(objects):
  raise Exception("shouldn't be here")
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
    with open('/Users/alaiasolko-breslin/Penn/research/blackbox-learning/scene/scene/llm.pkl', 'wb') as f:
      pickle.dump(queries, f)
    return ans
  raise Exception("LLM failed to provide an answer") 

def classify_llm_single(objects):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  answer = call_llm(objects)
  counts = torch.zeros(9)
  for a in answer:
    s = parse_response(a)
    if s in random_scene:
      counts[random_scene.index(s)] += 1
  return random_scene[counts.argmax()]

def call_llm_single(objects):
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

def parse_response(answer):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  for s in random_scene:
    if s in answer: return s
  raise Exception("LLM failed to provide an answer") 
