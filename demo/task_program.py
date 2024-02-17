def sum_2(xa, xb):
  return xa + xb

def sum_3(xa, xb, xc):
  return xa + xb + xc

def sum_4(xa, xb, xc, xd):
  return xa + xb + xc + xd

def classify_11(margin, shape, texture):
  if margin == 'serrate': return 'Ocimum basilicum'
  elif margin == 'indented': return 'Jatropha curcas'
  elif margin == 'lobed': return 'Platanus orientalis'
  elif margin == 'serrulate': return "Citrus limon"
  elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'
    elif shape == 'lanceolate': return 'Mangifera indica'
    elif shape == 'oblong': return 'Syzygium cumini'
    elif shape == 'obovate': return "Psidium guajava"
    else:
      if texture == 'aristate': return "Alstonia Scholaris"
      elif texture == 'round': return "Terminalia Arjuna"
      elif texture == 'glossy': return "Citrus limon"
      else: return "Punica granatum"
  else:
    if shape == 'elliptical': return 'Terminalia Arjuna'
    elif shape == 'lanceolate': return "Mangifera indica"
    else: return 'Syzygium cumini'

l11_margin = ["entire", "serrate", "lobed", 'indented', 'undulate', 'serrulate']
l11_shape = ["ovate", "oblong", "elliptical", 'obovate', 'lanceolate']
l11_texture = ["aristate", "round", 'glossy', "medium"]
l11_labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']
l11_dim = 2304