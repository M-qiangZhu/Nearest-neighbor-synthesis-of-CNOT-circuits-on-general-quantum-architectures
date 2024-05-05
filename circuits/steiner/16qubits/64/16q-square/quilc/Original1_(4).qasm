// EXPECTED_REWIRING [0 5 1 3 4 2 7 6 8 10 9 11 12 13 14 15]
// CURRENT_REWIRING [0 8 2 3 1 7 14 12 9 15 10 13 6 11 4 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[9];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122044*pi) q[15];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-3.075736053375836*pi) q[7];
rz(1.674236967864048*pi) q[8];
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
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[11];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.21353272314408345*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.077989633526895*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.381184772407102*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.9771162250494786*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.9581089657343345*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.6015542728903494*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(3.1339800237248836*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.50377955877053*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.28903648308757335*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.7221088860947344*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(1.9770105968746323*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.976207047482915*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(-2.087802470758894*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3844841619731474*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.2762476260936904*pi) q[6];
rz(-1.8598328098824712*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.5576213692594538*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.962804673790667*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.6812815023071153*pi) q[5];
rz(-1.9066621391296632*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7405105578528652*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5965918961117467*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
cz q[10], q[11];
rz(0.4385818901832407*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.9493897707586219*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.642390846176108*pi) q[6];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(0.39534911971118364*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.064669455657355*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.7661016573241899*pi) q[5];
rz(1.0537901828308982*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3844841619731452*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.4790089909250859*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.4361413542909993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6665530430790074*pi) q[13];
rx(3.141592653589793*pi) q[13];
rz(-2.6625757902999436*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8385954038498081*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.387104966695441*pi) q[2];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687789*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(-1.7843290499389812*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.8103884456122044*pi) q[11];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-0.6542456812873574*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970194*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(2.217663268376243*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.0384694512530912*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.1984242606411986*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.2054098339923036*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.3843276666658482*pi) q[9];
cz q[9], q[8];
rz(-0.9783302174689581*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-0.41122480169222*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674753*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.003614218325528*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.0779896335268955*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.3811847724071016*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[11], q[4];
rz(0.09296162521356113*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.8238268846775596*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.3361998643843176*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8134691601831683*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.8028997532304736*pi) q[6];
cz q[6], q[5];
rz(-0.7076873050771515*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.5811492890180702*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.7220797178950408*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.0287425718523773*pi) q[9];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.7843290499389812*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526896*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.8103884456122044*pi) q[14];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-2.3918013978459243*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.357263603650812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-2.6625757902999436*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8385954038498077*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.387104966695441*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[14], q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.9105968185919933*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.400611310044618*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(2.4873469723024364*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.217366411692774*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.003614218325527*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.0636030200628974*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.7604078811826915*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-2.851071111500697*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.38681006842452165*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5740348711458803*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.4054421423852275*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(3.141592653589793*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[13];
rz(2.761369489712264*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.9641888827222767*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.9438241621069082*pi) q[14];
rz(0.2427132517316299*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.2615998376377675*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.6971207428145691*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.18348368785545793*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.5400383806994444*pi) q[11];
rz(2.1295936339315267*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.384484161973147*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.2827859074684878*pi) q[12];
cz q[12], q[11];
rz(1.6366529270088535*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(-2.5642580454200985*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-2.8369564134151446*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.958108965734335*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.474181780485485*pi) q[5];
rz(0.7769136871373901*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-2.894265218789053*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.422277359048906*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5366205441837708*pi) q[6];
rz(-1.960585948106708*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.4557599579142788*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.1331514781308027*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.966913620713391*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.7705398111083106*pi) q[14];
cz q[14], q[13];
rz(2.0445531044894825*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(2.508167169514203*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.32128000598407297*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.1838196833234338*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.6957415579162085*pi) q[14];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(3.001876885502152*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.1645820567151595*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.1653856061068779*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.5146654427613733*pi) q[7];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.407787998523501*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6338545592296996*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.338047577851591*pi) q[13];
rz(2.217146941469614*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[5];
rz(-2.8988794018581627*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.8799928159520243*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[1];
cz q[5], q[2];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.3572636036508112*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8103884456122045*pi) q[3];
rz(-0.6734730460682394*pi) q[0];
rx(3.141592653589793*pi) q[0];
cz q[0], q[1];
rz(0.24271325173163089*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.261599837637768*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(1.674236967864048*pi) q[2];
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
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687783*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5146654427613733*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
cz q[7], q[6];
rz(-0.647503909837804*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.1280349963569294*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.050117348466655*pi) q[8];
rz(2.5479125518443664*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.18348368785545902*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6674108731043153*pi) q[6];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.3572636036508117*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.414329424690655*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731472*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.27624762609369*pi) q[9];
rz(1.7052786638315176*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.662919466807792*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(1.4564375502462912*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.426995486693993*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-3.059616298134544*pi) q[15];
rz(-1.1645820567151595*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1653856061068779*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[15];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-2.7640947867453742*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.754922226413092*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.6970893459204852*pi) q[9];
rz(2.217146941469613*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-0.8371083377258*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0454780355156306*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(1.2668772079426907*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(-3.074457494045478*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(2.217146941469614*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
rz(0.3039191188522059*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(0.6236520768094658*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.1593993372502314*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.1404237342881656*pi) q[10];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068779*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5146654427613733*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
rz(0.8371083377257977*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
rz(3.0744574940454803*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122044*pi) q[15];
rz(-2.3918013978459243*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.075736053375836*pi) q[9];
rx(1.5707963267948966*pi) q[13];
cz q[10], q[13];
cz q[4], q[5];
rz(2.3728638618809867*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3528026413932086*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.8045523727631614*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.4142310599628565*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.372046315587795*pi) q[10];
cz q[10], q[9];
rz(-1.740738306368793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-2.8988794018581627*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8799928159520252*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-2.244269372863135*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.9078768599871037*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-2.547912551844376*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.958108965734335*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6015542728903478*pi) q[5];
rz(-1.252010809771474*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9583600752412265*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.620417405249855*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.17572301558274073*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.15466790824696403*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5932644425266178*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(-1.164582056715156*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687844*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-2.4314042307726473*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-2.0854617695562743*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.9771162250494758*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.9581089657343353*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.4741817804854827*pi) q[9];
rz(0.7816398310657818*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.8999746626860248*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.9846049165996376*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8455707951022124*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.11315019143885204*pi) q[10];
cz q[10], q[5];
rz(-1.09720987853931*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.0595192458438059*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.33994375565725593*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.4594972153324264*pi) q[5];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-3.0381520125206416*pi) q[8];
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
rz(2.699565285330644*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.2211717812910718*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.4100817298727097*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.6225936092217825*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.761369489712264*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.9641888827222767*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.9438241621069082*pi) q[7];
rz(-0.3997503926745856*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.6980361171498525*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.0883522325935442*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.4421464432704612*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.44021563617524917*pi) q[11];
cz q[11], q[4];
rz(-0.5175509457803793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[5];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(-0.2135327231440851*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526896*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.8103884456122047*pi) q[14];
rz(2.2759621053062955*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.0475493807863683*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.9741848021012927*pi) q[2];
rz(-1.511882324602581*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.2913791962168345*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(0.9360076659883885*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(2.1223162034331104*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.675200626276474*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(-2.960296975321197*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.081288753785004*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.691748536436854*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0813386745216598*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.0019270560246376078*pi) q[2];
cz q[2], q[1];
rz(0.38230879259035166*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.234438261473284*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3424575645677406*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.778845328492112*pi) q[3];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rx(-1.5707963267948966*pi) q[6];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.3918013978459243*pi) q[9];
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
rz(-0.24171432676996302*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.1832441131482887*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.6596456431279895*pi) q[15];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rz(1.456437550246291*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4269954866939933*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.41856786008126595*pi) q[7];
cz q[0], q[7];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-0.8933906867100481*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.077438930057857*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.4466440503276825*pi) q[1];
rz(-2.393665013547763*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.9427107054770762*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[5];
rz(-0.3789880873412219*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068779*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5146654427613733*pi) q[10];
rz(-0.12325930933897668*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.7725444373124293*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.537459676619504*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5418230760072893*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.7427759459536967*pi) q[5];
cz q[5], q[2];
rz(-0.27390148877827336*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(0.7143733737146464*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.479463176524813*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.1374802577353474*pi) q[5];
rz(-0.6449675804976298*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[13];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9159160483656663*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[14];
cz q[15], q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
cz q[7], q[6];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[10];
rz(0.7792281771905801*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.37538183801786*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.9324969005150209*pi) q[2];
rz(-2.4281032659654596*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7135584132377142*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[2], q[5];
rx(-1.5707963267948966*pi) q[2];
rz(1.1765892518115226*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(-3.00841526963106*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.7720824204075676*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-0.6542456812873576*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9242262418970197*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.9244457121201792*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(0.05267051292929693*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.18348368785545804*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.540038380699441*pi) q[8];
rz(-1.0219540935245999*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.535882314707925*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.0366206373770916*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-0.6366625159348197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.4564375502462912*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.4269954866939927*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-1.488819971339647*pi) q[15];
cz q[15], q[8];
rx(1.5707963267948966*pi) q[14];
cz q[15], q[14];
rz(-2.1363610972068345*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.809320581443244*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(0.21353272314408317*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.0636030200628976*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.7604078811826918*pi) q[14];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(2.824752933336865*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.18348368785545835*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.540038380699442*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-1.5526354762711336*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6463506146747164*pi) q[6];
rz(2.065553122491837*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.8331259921626974*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.744022044098137*pi) q[2];
cz q[5], q[6];
rz(3.141592653589793*pi) q[6];
rz(-2.3918013978459243*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.9280599304457082*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268964*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.381184772407101*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
cz q[0], q[1];
rz(3.048456767380889*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.18348368785545788*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.5400383806994427*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[6];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(1.9770105968746388*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.9762070474829154*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[10];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-0.0073950738572623465*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rz(2.217146941469614*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(-2.6269272108284154*pi) q[9];
rx(3.141592653589793*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[10];
rz(1.0448716753785967*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.6939683919334133*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.0504645618592445*pi) q[11];
rz(-1.1645820567151595*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.1653856061068779*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.5146654427613733*pi) q[12];
rz(-1.5707963267948966*pi) q[13];
rz(-1.1645820567151595*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1653856061068779*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];