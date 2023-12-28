def sum_2(xa, xb):
  return xa + xb

def sum_3(xa, xb, xc):
  return xa + xb + xc

def sum_4(xa, xb, xc, xd):
  return xa + xb + xc + xd

def classify_plantvillage(margin, shape, apex):
  if margin == 'entire':
    if shape == 'cordate':
      return "potato"
    else:
      if apex == 'aristate':
        return "pepper"
      elif apex == 'round':
        return "soybean"
      else:
        return "blueberry"
  elif margin == 'serrate':
    if shape == 'cordate': 
      return "raspberry"
    elif shape == 'heart':
      return 'grape'
    else: 
      if apex == 'half':
        return "cherry"
      else:
        return 'apple'
  elif margin == 'lobbed':
    return "tomato"
  elif margin == 'mixed':
    return 'strawberry'
  else:
    return "peach"