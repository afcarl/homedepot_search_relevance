def process(file1, file2):
    with open(file2,'w') as fwrite:
        with open(file1,'r') as fread:
            for num,line in enumerate(fread.readlines()):
                if num ==0:
                    fwrite.write(line)
                else:
                    content = line.split(',')
                    key = content[0]
                    val = content[1].strip('\n')
                    if float(val) > 3.0:
                        val = 3.0
                        fin_line = key+','+str(val)+'\n'
                    else:
                        fin_line = key+','+str(val)+'\n'
                    fwrite.write(fin_line)




if __name__ == '__main__':
    process('single_0303_rmse4_2.csv','single_0303_rmse4_2_final.csv')


