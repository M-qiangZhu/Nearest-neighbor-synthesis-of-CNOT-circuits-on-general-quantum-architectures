OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[27];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[4];
cx q[4],q[1];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[21],q[23];
cx q[18],q[21];
cx q[15],q[18];
cx q[12],q[15];
cx q[10],q[12];
cx q[7],q[10];
cx q[4],q[7];
cx q[1],q[4];
cx q[0],q[1];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[7],q[4];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[8],q[11];
cx q[5],q[8];
cx q[3],q[5];
cx q[2],q[3];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[8],q[5];
cx q[11],q[8];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[12];
cx q[7],q[10];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[8],q[11];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[11],q[8];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[12];
cx q[7],q[10];
cx q[6],q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[8],q[11];
cx q[9],q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
cx q[18],q[21];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[18],q[15];
cx q[21],q[18];
cx q[23],q[21];
cx q[24],q[23];
cx q[25],q[24];
cx q[22],q[25];
cx q[19],q[22];
cx q[16],q[19];
cx q[14],q[16];
cx q[13],q[14];
cx q[15],q[18];
cx q[18],q[21];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[16],q[14];
cx q[19],q[16];
cx q[22],q[19];
cx q[25],q[22];
cx q[24],q[25];
cx q[23],q[24];
cx q[21],q[23];
cx q[18],q[21];
cx q[15],q[18];
cx q[16],q[19];
cx q[19],q[16];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[22],q[19];
cx q[25],q[22];
cx q[24],q[25];
cx q[23],q[24];
cx q[21],q[23];
cx q[18],q[21];
cx q[17],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[23],q[21];
cx q[24],q[23];
cx q[25],q[24];
cx q[22],q[25];
cx q[19],q[22];
cx q[20],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[22],q[25];
cx q[25],q[22];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
measure q[23] -> meas[23];
measure q[24] -> meas[24];
measure q[25] -> meas[25];
measure q[26] -> meas[26];
