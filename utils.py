# Run multiprocessing
import multiprocessing as mp
import traceback


def run_pll(f, args, processes=40):
    print('Multiprocessing {0} in {1} tasks, with {2} processes...'.format(f, len(args), processes))

    res_list = []
    error_list = []

    def log_result(res):
        res_list.append(res)

    def log_error(error):
        try:
            raise error
        except Exception:
            print(traceback.format_exc())
        error_list.append(error)

    pool = mp.Pool(processes=processes)
    for arg in args:
        pool.apply_async(f, kwds=arg, callback=log_result, error_callback=log_error)
    pool.close()
    pool.join()

    if len(error_list):
       print(f'{len(error_list)} jobs failed. ')

    print('Multiprocessing finished.')
    return res_list