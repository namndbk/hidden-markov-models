import string

punct = string.punctuation

punct = list(punct)  
tags = set()



print(punct)
f = open('test.txt')
for line in f:
	if line.strip():
		tokens = line.strip().split()
		if tokens[-1] != tokens[-2]:
			print(line)

