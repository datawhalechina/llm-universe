tasks=[]
def add_task(tasks,task):
    tasks.append(task)
    print(f"已添加：{task}")

def show_tasks(tasks):
    for index,item in enumerate(tasks,start=1):
        print(f"{index}. {item}")
while True:
    print("\n1.添加任务")
    print("2.查看任务")
    print("3.退出")

    choice=input("请选择:")
    if choice=="1":
        task_content=input("输入任务:")
        add_task(tasks,task_content)
    elif choice=="2":
        show_tasks(tasks)
    elif choice=="3":
        print("再见")
        break
    else:
        print("输入无效，请重新选择！")

