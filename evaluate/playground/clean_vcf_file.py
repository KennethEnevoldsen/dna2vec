import pandas as pd

def parse_all_sv_from_vcf(filepath):
    records = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            chrom, pos, vid, ref, alt, qual, filt, info = parts[:8]
            info_dict = {
                k: v for k, v in (
                    field.split('=') if '=' in field else (field, True)
                    for field in info.split(';')
                )
            }
            end = int(info_dict.get("END", int(pos)))
            svlen = info_dict.get("SVLEN", end - int(pos))
            if isinstance(svlen, list):
                svlen = svlen[0]
            try:
                svlen = int(svlen)
            except:
                svlen = "."
            svtype = info_dict.get("SVTYPE", ".")
            
            if svtype not in ["INS"]:
                continue
            
            info_str = f"SVTYPE={svtype};END={end};SVLEN={svlen}"
            records.append({
                "CHROM": chrom,
                "POS": int(pos),
                "ID": ".",
                "REF": ".",
                "ALT": f"<{svtype}>",
                "QUAL": qual,
                "FILTER": "PASS",
                "INFO": info_str
            })
    return pd.DataFrame(records)

def save_to_vcf(df, output_path):
    with open(output_path, 'w') as f:
        # VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        # VCF body
        for _, row in df.iterrows():
            f.write(f"{row['CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t{row['QUAL']}\t{row['FILTER']}\t{row['INFO']}\n")

# File paths
input_vcf_path = "/home/mehmet/codebase/dna2vec/HG002_SVs_Tier1_v0.6.vcf"
output_vcf_path = "/home/mehmet/codebase/dna2vec/HG002_SVs_Tier1_v0.6_cleaned.vcf"

# Parse and convert
all_sv_df = parse_all_sv_from_vcf(input_vcf_path)
save_to_vcf(all_sv_df, output_vcf_path)

print(f"All SVs saved to {output_vcf_path}")