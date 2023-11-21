
def subsample_shorter_reads(path="/mnt/SSD2/pholur/dna2vec/External_Data/", 
                            file_name = "hg002-oxford_nanopore.fastq", 
                            length=250, 
                            number_of_reads=1000):

    record_number = 0
    total_count = 0
    with open(path + file_name) as input:
        with open(f"{path}sample_{length}_{file_name}", "w") as output:
            for i,line2 in enumerate(input):
                if ".fastq" in file_name:
                    if i%4 == 1:
                        if record_number % 10 == 0:
                                total_count += 1
                                output.write(line2[:length] + '\n')
                        record_number += 1

                        if total_count > number_of_reads:
                            break
                else:
                    if record_number % 10 == 0:
                            total_count += 1
                            output.write(line2[:length] + '\n')
                    record_number += 1

                    if total_count > number_of_reads:
                        break


# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-oxford_nanopore.fastq", 250)
# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-oxford_nanopore.fastq", 500)
# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-oxford_nanopore.fastq", 1200)

# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-pacbio.fastq", 250)
# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-pacbio.fastq", 500)
# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "hg002-pacbio.fastq", 1200)

# subsample_shorter_reads("/mnt/SSD2/pholur/dna2vec/External_Data/", "illumina_wgs.fastq", 250)

subsample_shorter_reads("/mnt/SSD5/pholur/dna/", "reads.txt", 250)
subsample_shorter_reads("/mnt/SSD5/pholur/dna/", "reads.txt", 350)
subsample_shorter_reads("/mnt/SSD5/pholur/dna/", "reads.txt", 500)