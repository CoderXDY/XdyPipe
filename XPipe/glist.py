if point % 10 == 0:
    mem = psutil.virtual_memory()
    swp = psutil.swap_memory()
    cpu = psutil.cpu_times()
    netio = psutil.net_io_counters()
    pid = os.getpid()
    p = psutil.Process(pid)
    logger.error("record-" + str(point) + "....")
    logger.error(str(cpu))
    logger.error(str(mem))
    logger.error(str(swp))
    logger.error(str(netio))
    logger.error("process status:" + str(p.status()))
    logger.error(str(p.cpu_times()))
    logger.error(str(p.memory_info()))
point += 1