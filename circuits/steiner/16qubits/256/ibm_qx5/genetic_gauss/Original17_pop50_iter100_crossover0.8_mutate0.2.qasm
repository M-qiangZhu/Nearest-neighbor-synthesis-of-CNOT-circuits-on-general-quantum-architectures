// Initial wiring: [10, 12, 13, 15, 6, 11, 9, 4, 5, 14, 3, 7, 1, 2, 0, 8]
// Resulting wiring: [10, 12, 13, 15, 6, 11, 9, 4, 5, 14, 3, 7, 1, 2, 0, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[3], q[1];
cx q[4], q[2];
cx q[4], q[1];
cx q[5], q[4];
cx q[5], q[2];
cx q[5], q[0];
cx q[6], q[3];
cx q[6], q[2];
cx q[6], q[0];
cx q[7], q[5];
cx q[7], q[2];
cx q[8], q[7];
cx q[8], q[3];
cx q[8], q[0];
cx q[9], q[8];
cx q[9], q[7];
cx q[9], q[6];
cx q[9], q[0];
cx q[10], q[5];
cx q[10], q[3];
cx q[11], q[10];
cx q[11], q[8];
cx q[11], q[3];
cx q[8], q[1];
cx q[12], q[9];
cx q[12], q[7];
cx q[13], q[11];
cx q[13], q[7];
cx q[13], q[1];
cx q[14], q[12];
cx q[14], q[11];
cx q[14], q[10];
cx q[14], q[9];
cx q[15], q[12];
cx q[15], q[10];
cx q[15], q[1];
cx q[10], q[0];
cx q[12], q[3];
cx q[11], q[4];
cx q[9], q[5];
cx q[9], q[6];
cx q[13], q[8];
cx q[14], q[15];
cx q[13], q[14];
cx q[14], q[13];
cx q[12], q[13];
cx q[11], q[12];
cx q[10], q[12];
cx q[9], q[13];
cx q[9], q[11];
cx q[9], q[10];
cx q[10], q[9];
cx q[8], q[12];
cx q[12], q[8];
cx q[7], q[11];
cx q[6], q[12];
cx q[6], q[11];
cx q[6], q[10];
cx q[9], q[15];
cx q[12], q[14];
cx q[5], q[10];
cx q[5], q[9];
cx q[4], q[14];
cx q[4], q[13];
cx q[4], q[10];
cx q[4], q[9];
cx q[3], q[14];
cx q[3], q[11];
cx q[2], q[15];
cx q[2], q[13];
cx q[2], q[11];
cx q[2], q[10];
cx q[2], q[6];
cx q[2], q[5];
cx q[2], q[4];
cx q[4], q[2];
cx q[1], q[14];
cx q[1], q[12];
cx q[1], q[11];
cx q[1], q[10];
cx q[1], q[9];
cx q[1], q[6];
cx q[1], q[4];
cx q[0], q[12];
cx q[0], q[6];
cx q[0], q[4];
cx q[4], q[0];
cx q[6], q[8];
cx q[1], q[7];
