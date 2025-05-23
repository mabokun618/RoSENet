import numpy as np


def log_result(oa_ae, aa_ae, kappa_ae, element_acc_ae, path):
    f = open(path, 'w')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n' + '\n'
    f.write(sentence9)
    element_mean = list(element_mean)
    element_mean.extend([np.mean(oa_ae),np.mean(aa_ae),np.mean(kappa_ae)])
    element_std = list(element_std)
    element_std.extend([np.std(oa_ae),np.std(aa_ae),np.std(kappa_ae)])
    sentence10 = "All values without std: " + str(element_mean) + '\n' + '\n'
    f.write(sentence10)
    sentence11 = "All values with std: "
    for i,x in enumerate(element_mean):
        sentence11 += str(element_mean[i]) + " ± " +  str(element_std[i]) + ", "
    sentence11 += "\n"
    f.write(sentence11)
    f.close()