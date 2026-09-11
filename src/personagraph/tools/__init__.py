"""模型可调用能力、Tool Platform 内核与具体工具族。

公共类型由其 canonical 职责模块或窄 package facade 导出。根包刻意不 eager import
Catalog 持久控制面、执行器或具体工具，避免任意 ``personagraph.tools.*`` 导入产生
无关启动副作用。
"""
