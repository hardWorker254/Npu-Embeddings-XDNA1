00000000 <_Z12softmax_implILb0EEvP8bfloat16S1_i>:
       0:      	mova	r2, #0x0;		nopb	;		nopxm	
       c:      	ge	 r1, r2, r0
      10:      	jnz	 r1, #0x0
      16:      	nop	
      18:      	nop	
      1a:      	nop	
      1c:      	event	#0
      20:      	paddxm	 [sp], #0x400
      26:      	movxm	r1, #-0xeb6
      2c:      	vbcst.32	 x4, r2
      30:      	vbcst.16	 x6, r2
      34:      	movxm	r3, #-0x3d380000
      3a:      	mov	p2, sp
      3e:      	vbcst.16	 x0, r1
      42:      	movxm	r1, #0x3f80
      48:      	vbcst.32	 x3, r3
      4c:      	movxm	r3, #0x3fb8aa3b
      52:      	vbcst.16	 x2, r1
      56:      	vbcst.32	 x1, r3
      5a:      	movxm	r3, #-0x3d100000
      60:      	vmov	bmll0, x3
      64:      	vmov	bmlh0, x3
      68:      	vbcst.32	 x10, r3
      6c:      	movxm	r3, #-0xeb60d36
      72:      	vst	 bmll0, [sp, #-0x400];		vmov	bmll1, x1
      7a:      	vst	 bmlh0, [sp, #-0x3c0];		vmov	bmlh1, x1
      82:      	vst	 bmll1, [sp, #-0x300];		vmov	cmh0, cml0
      8a:      	mova	r5, #0x3c;		vst	 bmlh1, [sp, #-0x2c0];		movx	r4, #0x33c;		vmov	bmll2, x10
      96:      	mova	r7, #0x20;		vst	 bmhl0, [sp, #-0x380];		movx	r6, #0x10;		vmov	bmlh2, x10
      a2:      	mova	r17, #0xb;		vst	 bmhh0, [sp, #-0x340];		movx	r26, #0x8;		vbcst.32	 x8, r3
      ae:      	mova	r19, #0x7;		vst	 bmll2, [sp, #-0x280];		movx	r18, #0x9;		vmov	cmh2, cml2
      ba:      	mova	r20, #0x5;		vst	 bmlh2, [sp, #-0x240];		or	 r3, r2, r2;		vmov	lfh0, x8
      c6:      	mova	r22, #0x0;		vst	 bmhl2, [sp, #-0x200];		movx	r21, #0x40;		vbcst.64	 x8, r3:r2
      d2:      	mova	r3, #0x1;		paddb	 [p2], #-0x80;		movx	r1, #0x4;		vmov	lfh1, x8;		vst	 bmhh2, [sp, #-0x1c0]

000000e0 <.LBB1_2>:
      e0:      	nopa	;		lshl	 r23, r22, r3;		nopm	
      ea:      	movs	p3, p0;		lshl	 r24, r2, r3;		mov	m0, r23
      f4:      	padda	 [p3], m0;		mov	dj0, r24
      fa:      	vldb	 wl4, [p3, dj0]
		...
     10a:      	vmax_lt.bf16	 x5, r16, x4, x0
     10e:      	add	r24, r2, #0x10
     112:      	lshl	 r16, r24, r3
     116:      	mov	dj0, r16
     11a:      	vldb	 wl4, [p3, dj0]
		...
     12a:      	vmax_lt.bf16	 x5, r16, x4, x0
     12e:      	vmul.f	dm3, x5, x2, r4
     132:      	add.nc	lc, r1, #-0x2;		vmul.f	dm4, x5, x2, r4
     13a:      	movxm	ls, #0x0
     140:      	nopa	;		nopb	;		nops	;		movxm	le, #0x0;		nopv	
     150:      	nopa	;		nopb	;		nops	;		nopx	;		vmov	bmll0, lfh0;		nopv	
     160:      	nopa	;		nopb	;		nops	;		nopx	;		vmov	x11, lfh0;		nopv	
     170:      	nopx	;		vmov	x7, bmll3;		vsub.f	dm3, dm0, dm3, r5
     17a:      	add	r24, r24, #0x10;		vmov	x8, bmll4

00000180 <.LBB1_3>:
     180:      	nopa	;		nopb	;		lshl	 r16, r24, r3;		nopm	
     18c:      	mov	dj0, r16
     190:      	vldb	 wl9, [p3, dj0]
     194:      	nop	
     196:      	vconv.bf16.fp32	 wl4, bmll3
     19a:      	nop	
     19c:      	vlt.bf16	 r16, x4, x6
     1a0:      	nop	
     1a2:      	vsel.32	 x11, x11, x7, r16
     1a6:      	vmax_lt.bf16	 x7, r16, x9, x0
     1aa:      	nop	
     1ac:      	vmul.f	dm4, x7, x2, r4
     1b0:      	nop	
     1b2:      	vsub.f	dm3, dm0, dm4, r5
     1b6:      	vmov	bmll0, x11
     1ba:      	nop	
     1bc:      	vmov	x7, x8

000001c0 <.L_LEnd2>:
     1c0:      	nopa	;		nopb	;		nops	;		add	r24, r24, #0x10;		vmov	x8, bmll4;		nopv	
     1d0:      	nopa	;		nopb	;		nopx	
     1d8:      	vconv.bf16.fp32	 wl4, bmll3
     1dc:      	nop	
     1de:      	vlt.bf16	 r16, x4, x6
     1e2:      	nop	
     1e4:      	vsel.32	 x7, x11, x7, r16;		vsub.f	dm0, dm0, dm4, r5
     1ec:      	vmov	bmll0, x7
		...
     1f8:      	vconv.bf16.fp32	 wl4, bmll0
     1fc:      	nop	
     1fe:      	vlt.bf16	 r16, x4, x6
     202:      	nop	
     204:      	vsel.32	 x5, x7, x8, r16
     208:      	vshift	x7, x5, x0, r7
     20c:      	vmov	bmll0, x5;		vsub.f	dm0, dm0, dm3, r5
     214:      	vmov	bmll3, x7
		...
     220:      	vconv.bf16.fp32	 wl4, bmll0
     224:      	nop	
     226:      	vlt.bf16	 r24, x4, x6
     22a:      	nop	
     22c:      	vsel.32	 x5, x5, x7, r24
     230:      	vshift	x7, x5, x0, r6
     234:      	vmov	bmll0, x5;		vsub.f	dm0, dm0, dm3, r5
     23c:      	vmov	bmll3, x7
		...
     248:      	vconv.bf16.fp32	 wl4, bmll0
     24c:      	nop	
     24e:      	vlt.bf16	 r24, x4, x6
     252:      	nop	
     254:      	vsel.32	 x5, x5, x7, r24
     258:      	vshift	x7, x5, x0, r26
     25c:      	vmov	bmll0, x5;		vsub.f	dm0, dm0, dm3, r5
     264:      	vmov	bmll3, x7
		...
     270:      	vconv.bf16.fp32	 wl4, bmll0
     274:      	nop	
     276:      	vlt.bf16	 r24, x4, x6
     27a:      	nop	
     27c:      	vsel.32	 x5, x5, x7, r24
     280:      	vshift	x7, x5, x0, r1
     284:      	vmov	bmll0, x5;		vsub.f	dm0, dm0, dm3, r5
     28c:      	vmov	bmll3, x7
		...
     298:      	vlda	 bmll1, [sp, #-0x280];		vconv.bf16.fp32	 wl4, bmll0
     29e:      	add.nc	lc, r1, #0x0
     2a2:      	vlda	 bmlh1, [sp, #-0x240];		vlt.bf16	 r24, x4, x6
     2a8:      	movxm	ls, #0x0
     2ae:      	vlda	 bmhl1, [sp, #-0x200];		vsel.32	 x5, x5, x7, r24
     2b4:      	vlda	 bmhh1, [sp, #-0x1c0];		vst	 bmlh0, [sp, #-0x140];		vextbcst.32	 x5, x5, #0x0
     2be:      	vlda	 bmll4, [sp, #-0x300];		vst	 bmhl0, [sp, #-0x100];		vmov	bmll0, x5
     2c8:      	vlda	 bmlh4, [sp, #-0x2c0];		vst	 bmhh0, [sp, #-0xc0];		movxm	le, #0x0
     2d4:      	mova	r24, #0x0;		vst	 bmll0, [sp, #-0x180];		movx	r25, #0x0;		vmov	bmll3, lfh1

000002e0 <.LBB1_5>:
     2e0:      	lshl	 r16, r24, r3
     2e4:      	movs	dj0, r16
     2e8:      	vldb	 wl11, [p3, dj0]
		...
     2f8:      	vlda	 bmll2, [sp, #-0x180];		vmax_lt.bf16	 x8, r16, x11, x0
     2fe:      	vlda	 bmlh2, [sp, #-0x140]
     302:      	vlda	 bmhl2, [sp, #-0x100];		vmul.f	dm0, x8, x2, r4
     30a:      	nop	
     30c:      	vlda	 bmhh2, [sp, #-0xc0]
     310:      	nop	
     312:      	vlda	 bmll2, [sp, #-0x400]
     316:      	vlda	 bmlh2, [sp, #-0x3c0]
     31a:      	vlda	 bmhl2, [sp, #-0x380];		vmov	bmlh0, bmll0
     320:      	vlda	 bmhh2, [sp, #-0x340];		vsub.f	dm0, dm0, dm2, r5
     328:      	vmov	cmh0, cml0
     32c:      	nop	
     32e:      	nop	
     330:      	vsub.f	dm2, dm0, dm2, r5
		...
     33c:      	nop	
     33e:      	vconv.bf16.fp32	 wl4, bmll2
     342:      	nop	
     344:      	vlt.bf16	 r16, x4, x6
     348:      	vmov	x1, bmll0
     34c:      	vsel.32	 x5, x1, x3, r16
     350:      	vmov	bmll0, x5
     354:      	vconv.bf16.fp32	 x9, cml4
     358:      	vconv.bf16.fp32	 x7, cml0
     35c:      	vmsc.f	dm2, dm2, x9, x2, r5
     360:      	vmov	cml2, cml4;		vmsc.f	dm0, dm0, x7, x2, r5
		...
     370:      	vconv.bf16.fp32	 x8, cml2
     374:      	vconv.bf16.fp32	 x11, cml0
     378:      	vmsc.f	dm2, dm2, x8, x2, r5
     37c:      	vmsc.f	dm0, dm0, x11, x2, r5
		...
     388:      	vconv.bf16.fp32	 x5, cml2
     38c:      	vconv.bf16.fp32	 x1, cml0
     390:      	nop	
     392:      	vmul.f	dm0, x1, x5, r5
     396:      	vmul.f	dm2, x1, x8, r5
     39a:      	nop	
     39c:      	nop	
     39e:      	vadd.f	dm0, dm0, dm2, r5
     3a2:      	vmul.f	dm2, x11, x5, r5
     3a6:      	nop	
     3a8:      	nop	
     3aa:      	vadd.f	dm0, dm0, dm2, r5
     3ae:      	vmul.f	dm2, x7, x5, r5
     3b2:      	nop	
     3b4:      	nop	
     3b6:      	vadd.f	dm0, dm0, dm2, r5
     3ba:      	vmul.f	dm2, x11, x8, r5
     3be:      	nop	
     3c0:      	nop	
     3c2:      	vadd.f	dm0, dm0, dm2, r5
     3c6:      	vmul.f	dm2, x9, x1, r5
     3ca:      	nop	
     3cc:      	nop	
     3ce:      	vadd.f	dm0, dm0, dm2, r5
     3d2:      	vmul.f	dm2, x11, x9, r5
     3d6:      	nop	
     3d8:      	nop	
     3da:      	vadd.f	dm0, dm0, dm2, r5
     3de:      	vmul.f	dm2, x7, x8, r5
     3e2:      	nop	
     3e4:      	nop	
     3e6:      	vadd.f	dm0, dm0, dm2, r5
     3ea:      	vmul.f	dm2, x7, x9, r5
     3ee:      	nop	
     3f0:      	nop	
     3f2:      	vadd.f	dm0, dm0, dm2, r5
     3f6:      	nop	
     3f8:      	nop	
     3fa:      	vsub.f	dm2, dm0, dm1, r5
		...
     406:      	nop	
     408:      	vconv.bf16.fp32	 wl4, bmll2
     40c:      	nop	
     40e:      	vlt.bf16	 r16, x4, x6
     412:      	vmov	x7, bmll0
     416:      	vsel.32	 x7, x7, x10, r16
     41a:      	vmov	bmhh4, x7
     41e:      	nop	
     420:      	vexp2	 wl4, bmhh4
     424:      	nop	
     426:      	vmul.f	dm0, x4, x2, r4
		...
     432:      	nop	
     434:      	lshl	 r16, r25, r20;		vmov	bmlh0, bmll0
     43a:      	movs	dj0, r16;		add	r24, r24, #0x10

00000440 <.L_LEnd1>:
     440:      	nopa	;		nopb	;		vst	 wl4, [p2, dj0];		add	r25, r25, #0x1;		vmov	cmh0, cml0;		vadd.f	dm3, dm3, dm0, r5
     450:      	nopa	;		nopxm	
		...
     462:      	vmov	x5, bmll3;		vadd.f	dm0, dm3, dm0, r5
     46a:      	vshuffle	bmll0, x5, x5, r17
		...
     476:      	vmov	x5, bmll0;		vadd.f	dm0, dm0, dm3, r5
     47e:      	vshuffle	bmll3, x5, x5, r18
		...
     48a:      	vmov	x5, bmll0;		vadd.f	dm0, dm0, dm3, r5
     492:      	vshuffle	bmll3, x5, x5, r19
		...
     49e:      	vmov	x5, bmll0;		vadd.f	dm0, dm0, dm3, r5
     4a6:      	vshuffle	bmll3, x5, x5, r20
		...
     4b2:      	vmov	x8, bmll0
     4b6:      	vmov	x9, bmlh0
     4ba:      	vshuffle	x8, x8, x9, r1
     4be:      	vextract.32	 r16, x8, #0x0, vaddsign1
     4c2:      	nop	
     4c4:      	inv	 r16, r16
     4c8:      	add.nc	lc, r1, #0x0
     4cc:      	movxm	ls, #0x0
     4d2:      	movs	m0, r23;		movxm	le, #0x0
     4dc:      	mova	r23, #0x0;		movs	p3, p1;		vbcst.32	 x8, r16
     4e6:      	padda	 [p3], m0;		movx	r24, #0x0;		vmov	bmll3, x8

000004f0 <.LBB1_7>:
     4f0:      	nopa	;		nopb	;		lshl	 r16, r24, r20;		nopm	
     4fc:      	movs	dj0, r16
     500:      	vldb	 wl4, [p2, dj0]
		...
     510:      	vmul.f	dm4, x4, x2, r4
		...
     51c:      	nop	
     51e:      	vconv.bf16.fp32	 x8, cml4
     522:      	vconv.bf16.fp32	 x1, cml3
     526:      	vmsc.f	dm0, dm4, x8, x2, r5
     52a:      	vmsc.f	dm2, dm1, x1, x2, r5
     52e:      	vmov	cml1, cml3
     532:      	nop	
     534:      	nop	
     536:      	nop	
     538:      	vconv.bf16.fp32	 x5, cml0
     53c:      	vconv.bf16.fp32	 x7, cml2
     540:      	vmsc.f	dm4, dm0, x5, x2, r5
     544:      	vmsc.f	dm0, dm2, x7, x2, r5
		...
     550:      	vconv.bf16.fp32	 x9, cml4
     554:      	vconv.bf16.fp32	 x11, cml0
     558:      	nop	
     55a:      	vmul.f	dm1, x9, x11, r5
     55e:      	vmul.f	dm2, x9, x7, r5
     562:      	nop	
     564:      	vmul.f	dm4, x5, x11, r5
     568:      	vadd.f	dm1, dm1, dm2, r5
     56c:      	nop	
     56e:      	vmul.f	dm0, x8, x11, r5
     572:      	vadd.f	dm1, dm1, dm4, r5
     576:      	nop	
     578:      	vmul.f	dm2, x5, x7, r5
     57c:      	vadd.f	dm1, dm1, dm0, r5
     580:      	nop	
     582:      	vmul.f	dm4, x1, x9, r5
     586:      	vadd.f	dm1, dm1, dm2, r5
     58a:      	nop	
     58c:      	vmul.f	dm0, x5, x1, r5
     590:      	vadd.f	dm1, dm1, dm4, r5
     594:      	nop	
     596:      	vmul.f	dm2, x8, x7, r5
     59a:      	vadd.f	dm1, dm1, dm0, r5
     59e:      	nop	
     5a0:      	vmul.f	dm4, x8, x1, r5
     5a4:      	vadd.f	dm1, dm1, dm2, r5
     5a8:      	nop	
     5aa:      	nop	
     5ac:      	vadd.f	dm1, dm1, dm4, r5
     5b0:      	nop	
     5b2:      	nop	
     5b4:      	nop	
     5b6:      	lshl	 r16, r23, r3
     5ba:      	add	r24, r24, #0x1;		mov	dj0, r16

000005c0 <.L_LEnd0>:
     5c0:      	nopa	;		nopb	;		vst.conv.bf16.fp32	 bmll1, [p3, dj0];		add	r23, r23, #0x10;		nopm	;		nopv	
     5d0:      	nopa	;		nopb	;		add	r0, r0, #-0x1;		nopm	;		nops	
     5de:      	jnz	 r0, #0x0
		...
     5ec:      	add	 r22, r22, r21

000005f0 <.LBB1_9>:
     5f0:      	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
     600:      	nop	
     602:      	nop	
     604:      	nop	
     606:      	paddxm	 [sp], #-0x400
     60c:      	event	#1

Disassembly of section .text.softmax_poly_bf16:

00000000 <softmax_poly_bf16>:
       0:      	nopa	;		nopb	;		nops	;		j	#0x0;		nopv	
      10:      	nopa	;		nopx	
      16:      	nop	
      18:      	nop	
      1a:      	nop	
      1c:      	mova	r0, #0x40

Disassembly of section .text._Z12softmax_implILb1EEvP8bfloat16S1_i:

00000000 <_Z12softmax_implILb1EEvP8bfloat16S1_i>:
