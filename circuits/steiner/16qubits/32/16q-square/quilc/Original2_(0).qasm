// EXPECTED_REWIRING [0 1 2 3 4 5 7 6 8 9 10 11 13 12 14 15]
// CURRENT_REWIRING [3 15 0 1 11 13 7 9 8 12 14 2 6 4 5 10]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.59368010174542*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.958108965734335*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.4741817804854853*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(1.674236967864048*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.7843290499389812*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268964*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[10];
cz q[10], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-2.087802470758894*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.3844841619731474*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.2762476260936904*pi) q[11];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.7843290499389812*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.077989633526896*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.8103884456122044*pi) q[12];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[14];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[9], q[6];
rz(1.0537901828308989*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3844841619731472*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.27624762609369*pi) q[14];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.1034406410691524*pi) q[6];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(2.761369489712264*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9641888827222767*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.9438241621069082*pi) q[10];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[11];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
cz q[6], q[7];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687783*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7900023057508099*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.9236633008576913*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.7289078925588519*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rx(1.5707963267948966*pi) q[4];
rz(-1.130224223537565*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.4564375502462918*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.426995486693993*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-3.059616298134544*pi) q[12];
rz(3.062148276737266*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5395761164454771*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(0.749791255743869*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-2.6625757902999436*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8385954038498077*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.387104966695441*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(1.5707963267948966*pi) q[0];
cz q[2], q[3];
rz(1.3572636036508117*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.381184772407101*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[7], q[0];
rz(3.141592653589793*pi) q[8];
rz(1.2859356045296209*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.3509436796836002*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.5872252283822292*pi) q[0];
rz(-0.5293577000099374*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6820814028121251*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(3.0971150484546444*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
cz q[7], q[8];
rz(-0.8707747974068126*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.7868400828763336*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.731851539762527*pi) q[0];
rz(-1.7843290499389814*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.0636030200628965*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.09351401756525068*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.760407881182692*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(-2.5077911552736487*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.6084380878211304*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.980509525699814*pi) q[2];
rz(-1.8846312993447327*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-2.3918013978459243*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[8], q[7];
rz(2.761369489712264*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9641888827222767*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.9438241621069082*pi) q[9];
rz(-0.1034406410691524*pi) q[7];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687783*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.5146654427613733*pi) q[9];
rz(1.4564375502462918*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.426995486693993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.059616298134544*pi) q[10];
rz(0.3864974944028522*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.98246691639367*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.6489745765899453*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4782177318788055*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.746013656247049*pi) q[2];
cz q[2], q[1];
rz(-1.8840646879404837*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
cz q[10], q[9];
rz(0.10344064106915161*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.075736053375836*pi) q[9];
rz(0.4185649518560259*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.6683871612572863*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.813167445242296*pi) q[14];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122044*pi) q[15];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(0.13799790240498247*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.1102379544597767*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.6420126811149172*pi) q[2];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[2], q[5];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[14];
cz q[15], q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(-1.4553215971922162*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6073571083205147*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.8628920950574166*pi) q[1];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rz(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(0.749791255743869*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.784329049938982*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268955*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.381184772407101*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[10];
rz(-2.627378587204374*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.384484161973149*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.436141354291*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[11];
rz(-0.21920597895591243*pi) q[11];
cz q[6], q[1];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.487346972302436*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.2173664116927734*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[1];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(0.8271297992717612*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.6885596757449914*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.009281110175170859*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(2.4897477230283993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(2.5809980047382064*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
rz(0.7767051408062562*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
cz q[11], q[4];
cz q[2], q[1];
rz(-2.4322411164009154*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.1959887400036298*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.9156330120238823*pi) q[5];
rz(1.2400307164201383*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.958108965734335*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.474181780485484*pi) q[9];
rz(0.09415953089398164*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.784329049938982*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0779896335268955*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.9770105968746388*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.9762070474829154*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(2.217146941469614*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.495242038915076*pi) q[9];
rz(-2.1058337627385146*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.32675267390073165*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.9201328299945111*pi) q[2];
rz(-2.0016748667444855*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3591802730526683*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-2.33522918322013*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-1.164582056715162*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687789*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
cz q[9], q[8];
rz(-2.7640947867453742*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.754922226413092*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(0.6970893459204852*pi) q[15];
rz(1.5181258138655969*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.3572636036508126*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(2.238207199899204*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-1.0967567025069482*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.5226383254796283*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.5621273463506581*pi) q[2];
cz q[5], q[2];
rz(-0.3508373188672538*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(1.4564375502462925*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.426995486693993*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.7128745579459749*pi) q[7];
rz(-0.6542456812873577*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970202*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.1645820567151592*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687794*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.730367851897572*pi) q[5];
cz q[10], q[5];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747164*pi) q[0];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.495242038915076*pi) q[2];
rz(1.5360648544654631*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.876242869811959*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.9124058598902596*pi) q[3];
rz(3.141592653589793*pi) q[4];
rz(-0.1034406410691524*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(2.510694451099068*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(-2.495242038915075*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(1.056130884033526*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(1.4564375502462912*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.426995486693993*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.059616298134544*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(0.5146654427613782*pi) q[13];
rx(3.141592653589793*pi) q[13];
rz(-1.1645820567151595*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1653856061068779*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(-1.1645820567151592*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687805*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.62692721082842*pi) q[15];
