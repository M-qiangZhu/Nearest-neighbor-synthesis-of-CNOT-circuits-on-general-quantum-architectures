// Initial wiring: [3, 0, 9, 10, 2, 1, 13, 14, 8, 12, 15, 11, 5, 7, 6, 4]
// Resulting wiring: [3, 0, 9, 10, 2, 1, 13, 14, 8, 12, 15, 11, 5, 7, 6, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[4], q[3];
cx q[3], q[2];
cx q[4], q[3];
cx q[5], q[4];
cx q[6], q[1];
cx q[6], q[5];
cx q[1], q[0];
cx q[7], q[6];
cx q[6], q[1];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[7], q[6];
cx q[8], q[7];
cx q[9], q[8];
cx q[10], q[5];
cx q[11], q[4];
cx q[4], q[3];
cx q[11], q[4];
cx q[12], q[11];
cx q[11], q[4];
cx q[4], q[3];
cx q[11], q[4];
cx q[14], q[13];
cx q[13], q[10];
cx q[14], q[13];
cx q[14], q[15];
cx q[10], q[13];
cx q[9], q[10];
cx q[10], q[11];
cx q[8], q[9];
cx q[8], q[15];
cx q[9], q[10];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[9], q[8];
cx q[11], q[10];
cx q[6], q[9];
cx q[5], q[10];
cx q[5], q[6];
cx q[4], q[5];
cx q[4], q[11];
cx q[5], q[10];
cx q[5], q[6];
cx q[2], q[5];
cx q[5], q[10];
cx q[5], q[6];
cx q[10], q[13];
cx q[6], q[7];
cx q[1], q[6];
