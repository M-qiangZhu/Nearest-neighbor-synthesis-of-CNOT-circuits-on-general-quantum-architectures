// EXPECTED_REWIRING [2 0 1 3 4 10 6 14 7 9 11 5 12 13 15 8]
// CURRENT_REWIRING [14 2 1 7 9 10 6 13 0 5 3 12 4 11 8 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.7843290499389812*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8103884456122044*pi) q[3];
rz(0.59368010174542*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4741817804854853*pi) q[0];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[1], q[6];
rz(-1.784329049938982*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.0636030200628972*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.760407881182692*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-2.087802470758894*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.3844841619731474*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.2762476260936904*pi) q[4];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747164*pi) q[0];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[11], q[12];
rz(-1.1645820567151592*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.16538560610687794*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.730367851897572*pi) q[1];
cz q[1], q[6];
rz(0.10344064106915161*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.3572636036508121*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
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
rz(-2.8988794018581627*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8799928159520249*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.8052382068259782*pi) q[10];
cz q[9], q[10];
cz q[7], q[0];
rz(-3.0381520125206416*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.3572636036508117*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268964*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.4249210700633186*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3844841619731467*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.27624762609369*pi) q[10];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(-1.1645820567151568*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068771*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(0.9210270573549552*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7592861559529326*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5329385814081944*pi) q[6];
rz(-2.6724007151482176*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4290942443428525*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[6];
rz(0.6851288267188593*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-2.3918013978459243*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.2989944927003583*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268955*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-2.381184772407101*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.495242038915076*pi) q[2];
rz(2.6595014223059357*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.250211134965431*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.40422800204653*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0581652304950935*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.1431019121355117*pi) q[10];
cz q[10], q[5];
rz(2.8637036202308934*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[3];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.989035248778917*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6903820417170294*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(2.761369489712264*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.9641888827222767*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.9438241621069082*pi) q[11];
rz(-0.117945714916887*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5460298244653161*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.714641587172708*pi) q[4];
rz(-2.1755555664069646*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.0156970120711524*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-2.043425825437156*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.13913801885182342*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.3228188009871564*pi) q[6];
rz(-3.0210373283910443*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.8795965257078275*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(-1.5707963267948966*pi) q[6];
rz(1.2060987270093184*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(-1.7245740718780227*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.682640878229725*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.8657195172666381*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.565001727259962*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.6731408128173091*pi) q[11];
cz q[11], q[10];
rz(-1.4474332592613985*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-2.5479125518443744*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.958108965734335*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.6015542728903497*pi) q[13];
rz(-1.7843290499389812*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.381184772407101*pi) q[14];
cz q[14], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(1.5707963267948966*pi) q[4];
rz(-0.4449006852762558*pi) q[4];
rz(2.3338465820356573*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4438245214636263*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.4391411553720208*pi) q[11];
rz(-2.9280599304457064*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.0636030200628974*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.7604078811826911*pi) q[12];
cz q[12], q[11];
rz(-2.8769197687304153*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-2.7783054207918685*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.1110470146971245*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.6114003218517711*pi) q[9];
rz(0.23709428848438924*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.032417694702106*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.284289382304655*pi) q[13];
rz(2.676346135622607*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.5676642036422255*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(-1.5707963267948966*pi) q[13];
rz(-1.3821801688736155*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-0.6542456812873576*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.9242262418970197*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.7271534186710022*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.6038268161202546*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.760013854450666*pi) q[5];
rz(-1.855876528729967*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.182047939021104*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6234556116877492*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(2.9500419658760713*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4462491130816422*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.2902537281572697*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9919629676489968*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.7414046622362171*pi) q[13];
cz q[13], q[10];
rz(-1.855530815215106*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5301156954162214*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.176551245878368*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.1741127361547807*pi) q[14];
cz q[14], q[9];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(0.16007096702124657*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(2.7414167549061412*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.5639162496111334*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.2417207881366468*pi) q[6];
rz(-2.1083916526358943*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.6348637022141436*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[6];
rz(-0.5346004560222977*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-1.2717170792176158*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3925692168941923*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.6192598432508274*pi) q[3];
rz(-1.9107238022365336*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.312674591363405*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(-1.5707963267948966*pi) q[3];
rz(-2.996875365207205*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(2.217146941469614*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.4673556857257442*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.905363143570253*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.1482387681147355*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.438437518107853*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.32138573990936603*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.7646103709575387*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.6161769858325908*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-1.728140280606288*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6408233078748165*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.42655527048000863*pi) q[4];
rz(-2.044125564487894*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[5];
rz(-2.9837591636323717*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7925413657751483*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.3504201232787788*pi) q[10];
rx(1.5707963267948966*pi) q[11];
rz(-0.34711116552717236*pi) q[11];
rz(-2.394392698120301*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(1.0537901828308989*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731472*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.27624762609369*pi) q[9];
rx(-1.5707963267948966*pi) q[6];
rz(-0.2135327231440849*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
cz q[11], q[10];
rz(-0.8197991082805154*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6873971104770096*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.5834477111332985*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-2.57094571833896*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(0.23686827246525813*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.2189262831790817*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.0130114085503363*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.126396887564377*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.5470249067741249*pi) q[9];
cz q[9], q[6];
rz(1.0447395946195428*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(2.1943866184358227*pi) q[14];
rz(0.9053576356512929*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.670914087323049*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.012068351025646962*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.9774825824222428*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.1239820632006423*pi) q[6];
rz(0.9771162250494766*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.18348368785545824*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6674108731043078*pi) q[0];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[6];
rz(-2.0437444410562953*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.2421579598535075*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.6883508985934342*pi) q[9];
cz q[9], q[14];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5834477111332994*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(3.141592653589793*pi) q[6];
rz(2.454957390025829*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rz(1.3572636036508112*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526896*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.8103884456122045*pi) q[14];
rz(1.674236967864048*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.075736053375836*pi) q[10];
rz(-1.0537901828308989*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.757108491616646*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.8653450274961034*pi) q[11];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[8], q[7];
rz(-1.1275650072919743*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.6855992245371592*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.0474573256255708*pi) q[9];
rz(-0.48004627230320096*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7824496365034508*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[9];
rz(0.5186688710116858*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-3.0217545854078387*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4649316967103534*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.513657107487253*pi) q[0];
rz(1.544089606641904*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.9080321553979158*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(3.04004847649973*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.23870397664525597*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7256485975687249*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.0976858601477852*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9058828343564297*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.823373235595004*pi) q[7];
cz q[7], q[6];
rz(1.5721727165633812*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.7904011388040064*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7769631804617568*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.8382253655499614*pi) q[0];
rz(2.7620363576957816*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.9151541939430998*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.704656043642765*pi) q[7];
cz q[7], q[0];
rz(2.8865096513881454*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.594728742735498*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(0.22900145976058314*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.0155418269987664*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.761222112322935*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1729101187517733*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.6457932409866068*pi) q[15];
cz q[15], q[8];
rz(-0.5167448971743072*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
rz(-2.8363719554189313*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.5605087049871785*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.8016496095500913*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.1195366998752387*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.4943338647724409*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.8654119631230296*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(0.2427132517316307*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4681196075215537*pi) q[0];
rz(-1.2535072838815586*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.571762559065554*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.8352604038968181*pi) q[1];
rz(2.132794376869347*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[2];
rz(-0.8734418178388837*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.006224269036756*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.4532711814186454*pi) q[3];
rz(2.044125564487894*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rz(0.6900681920204229*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3220139815661132*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.918658293307465*pi) q[6];
rz(-0.1034406410691524*pi) q[7];
rz(-0.1960268409974546*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(0.6463506146747173*pi) q[9];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068779*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5146654427613733*pi) q[10];
rz(1.4564375502462912*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.426995486693993*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.059616298134544*pi) q[11];
rz(-1.1645820567151595*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.1653856061068779*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.5146654427613733*pi) q[12];
rz(-1.5707963267948966*pi) q[13];
rz(-2.9148370590839923*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.606955746890229*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.8180487969698573*pi) q[14];
rz(2.1644619411313295*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.6271058468740383*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.1368046359181765*pi) q[15];
