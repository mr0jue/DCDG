# # coding-utf-8
def grade():
	while True:
		sys_option_results=list(input('系统正确的答案：'))
		student_option_results=list(input("请输入学生的选项:"))
		if len(sys_option_results)==len(student_option_results):
			break
		pass
	grade=0
	for i in range(len(sys_option_results)):
		if sys_option_results[i]==student_option_results[i]:
			grade+=10
		else:
			grade+=0
	return grade
print(grade())
# print(sys_option_results)
# def generator(lenl):
# 	# student_option_results=student_option_results
# 	for a in student_option_results:
# 		# a+=1
# 		# for i in student_option_results:
# 		yield a
# 	for b in sys_option_results:
# 		yield b
# 	print(a,b)
# 	return (a,b)
# for n in generator(len(sys_option_results)+1):
# 	print('alkdfjlkad',n)
# print(generator())
# for j in sys_option_results:
# 	yield j
# if i==j:
# 	grade+=10
# else:
# 	grade=grade

