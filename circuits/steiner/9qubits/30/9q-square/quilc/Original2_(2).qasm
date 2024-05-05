// EXPECTED_REWIRING [0 1 2 3 4 5 7 8 6]
// CURRENT_REWIRING [0 8 2 7 3 5 6 4 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(0.59368010174542*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.958108965734335*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4741817804854853*pi) q[7];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[4];
rz(-2.687040225304508*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.180311834696913*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.1621516414273794*pi) q[3];
rz(0.6270827066078155*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.242797595429085*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[3], q[8];
rx(-1.5707963267948966*pi) q[3];
rz(1.7342804534561136*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[3], q[8];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[3], q[8];
rz(-2.1562080597884767*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.7625888637863478*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.3995440474919647*pi) q[3];
rz(1.357263603650812*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0779896335268964*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[4];
cz q[4], q[3];
rz(-2.0040107248383245*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-1.1326349595418248*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5832510640350301*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.11871949428460203*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.432037545286481*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-3.136623743471344*pi) q[8];
cz q[8], q[7];
rz(-0.8831181258354887*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(0.59368010174542*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4741817804854853*pi) q[0];
rz(1.9770105968746396*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.9762070474829154*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(-0.6357158078882993*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.6284601765973258*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.3215997263307276*pi) q[0];
rz(-2.55304780957376*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.2108147105424623*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(-1.5707963267948966*pi) q[0];
rz(1.9865975386087928*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.2989944927003605*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.063603020062897*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7604078811826913*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.0080334282152683*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.333214709348789*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.8855069361207137*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(-2.3918013978459243*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.347781588026086*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.2822755701418365*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.672937178136436*pi) q[8];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.4001075957864373*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.1645820567151595*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.1653856061068779*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.5146654427613733*pi) q[4];
rz(-2.858419245535124*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5803740588357458*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.5680093209155719*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-0.6733946557402944*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(2.9280599304457087*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.0779896335268964*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8103884456122046*pi) q[2];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
cz q[3], q[8];
rz(1.760527017009314*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.2437778563939483*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.2359908246819784*pi) q[7];
rz(2.3012990926249204*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-0.25600166006257297*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[4];
rz(-0.08415215115530561*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[7], q[4];
cz q[5], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.1645820567151595*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.1653856061068779*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.5146654427613733*pi) q[2];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[5];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
rz(1.8267979868574695*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.0381520125206407*pi) q[8];
rz(-2.3918013978459243*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.21353272314408464*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.3811847724071016*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.24271325173163086*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.261599837637768*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.8973232807266569*pi) q[1];
cz q[1], q[0];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
cz q[4], q[5];
rz(1.4673556857257446*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7226142745223187*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.213532723144086*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0779896335268955*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.5049397265809397*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.381184772407101*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.1645820567151595*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.1653856061068779*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.5146654427613733*pi) q[4];
rx(-1.5707963267948966*pi) q[3];
cz q[4], q[3];
rz(-2.0878024707588945*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3844841619731472*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.27624762609369*pi) q[8];
rz(2.547912551844373*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.18348368785545807*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6674108731043082*pi) q[4];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(1.674236967864048*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.21353272314408445*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.063603020062897*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7604078811826924*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.1775808061114201*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.082728699543747*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.756537446818116*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.9109491745758156*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.674236967864048*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(-2.830902119661549*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.5174806531557159*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.7876846429977675*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-2.9939540225827823*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(1.5707963267948966*pi) q[0];
rz(1.983653724828861*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[3], q[4];
rz(-1.6742369678640483*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.16458205671516*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068783*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.0561308840335233*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-1.164582056715158*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.1653856061068785*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(3.141592653589793*pi) q[0];
rz(0.2427132517316307*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.261599837637768*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.4681196075215537*pi) q[1];
rz(-1.1645820567151595*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.1653856061068779*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.5146654427613733*pi) q[2];
rz(2.217146941469614*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-2.085461769556271*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[7];
rx(3.141592653589793*pi) q[7];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.6269272108284194*pi) q[8];
