00000000 <_Z12softmax_implILb1EEvP8bfloat16S1_i>:
       0:      	mova	r2, #0x0;		nopb	;		nopxm	;		nops	
       e:      	ge	 r1, r2, r0
      12:      	jnz	 r1, #0x0
      18:      	nop	
      1a:      	nop	
      1c:      	nop	
      1e:      	event	#0
      22:      	paddxm	 [sp], #0xec0
      28:      	movxm	r1, #0x5301
      2e:      	movxm	r3, #0x3f317218
      34:      	vbcst.16	 x7, r1
      38:      	movxm	r1, #0x4b01
      3e:      	vbcst.16	 x5, r1
      42:      	movxm	r1, #-0xeb6
      48:      	vbcst.16	 x0, r1
      4c:      	movxm	r1, #0x3f80
      52:      	vbcst.16	 x2, r1
      56:      	movxm	r1, #-0x3d380000
      5c:      	vbcst.32	 x11, r1
      60:      	movxm	r1, #0x3fb8aa3b
      66:      	vbcst.32	 x1, r1
      6a:      	movxm	r1, #0x39244f63
      70:      	vmov	bmlh3, x11
      74:      	vbcst.32	 x8, r1
      78:      	movxm	r1, #0x3aaebe2f
      7e:      	vmov	bmll3, x1
      82:      	vmov	bmll0, x8
      86:      	vbcst.32	 x8, r1
      8a:      	movxm	r1, #0x3c1d8e5c
      90:      	vmov	bmll0, x8
      94:      	vst	 bmll0, [sp, #-0xbc0];		vbcst.32	 x8, r1
      9c:      	movxm	r1, #0x3d635854
      a2:      	vmov	bmll2, x8
      a6:      	vbcst.32	 x8, r1
      aa:      	movxm	r1, #-0x3d100000
      b0:      	vbcst.32	 x10, r1
      b4:      	vst	 bmlh3, [sp, #-0x980];		movxm	r1, #0x3781e215
      be:      	vst	 bmll3, [sp, #-0xa40];		vmov	bmll4, x8
      c6:      	vst	 bmll0, [sp, #-0x640];		vmov	bmlh0, bmll0
      ce:      	vst	 bmll0, [sp, #-0x600];		vbcst.32	 x8, r1
      d6:      	vst	 bmll2, [sp, #-0x540];		movxm	r1, #0x3e75fe06
      e0:      	vst	 bmll2, [sp, #-0x500];		vmov	bmlh1, x10
      e8:      	vst	 bmll4, [sp, #-0x440];		vmov	bmll1, x10
      f0:      	vst	 bmll4, [sp, #-0x400];		vmov	bmlh2, bmll2
      f8:      	vst	 bmll1, [sp, #-0x8c0];		vmov	bmll3, x8
     100:      	vst	 bmlh1, [sp, #-0x880];		vmov	cmh0, cml0
     108:      	vst	 bmll3, [sp, #-0xac0];		vbcst.32	 x8, r1
     110:      	vst	 bmhl0, [sp, #-0x5c0];		vmov	cmh1, cml1
     118:      	vst	 bmhh0, [sp, #-0x580];		vmov	bmlh4, bmll4
     120:      	vst	 bmhl1, [sp, #-0x840];		vmov	cmh2, cml2
     128:      	vst	 bmhh1, [sp, #-0x800];		vmov	bmll3, x8
     130:      	vlda	 bmll3, [sp, #-0xa40];		vst	 bmhl2, [sp, #-0x4c0];		vbcst.32	 x8, r3
     13a:      	vst	 bmhh2, [sp, #-0x480];		vmov	cmh4, cml4
     142:      	vst	 bmll3, [sp, #-0xcc0];		movxm	r3, #0x3f800000
     14c:      	vlda	 bmll1, [sp, #-0xbc0];		vst	 bmhl4, [sp, #-0x3c0];		vmov	bmll3, x8
     156:      	vst	 bmhh4, [sp, #-0x380];		vmov	bmll3, x11
     15e:      	vst	 bmll3, [sp, #-0xdc0];		vbcst.32	 x8, r3
     166:      	vst	 bmll3, [sp, #-0x9c0];		vmov	cmh3, cml3
     16e:      	vst	 bmll3, [sp, #-0xa40];		vmov	bmll3, x8
     176:      	vst	 bmll3, [sp, #-0xa00]
     17a:      	vlda	 bmll3, [sp, #-0xac0];		vst	 bmll3, [sp, #-0xec0]
     180:      	vlda	 bmll1, [sp, #-0xec0];		vst	 bmll1, [sp, #-0xbc0];		vmov	bmlh1, bmll1
     18a:      	vlda	 bmll3, [sp, #-0xcc0];		vst	 bmll1, [sp, #-0xb80]
     190:      	vst	 bmhl3, [sp, #-0x940];		vmov	cmh1, cml1
     198:      	vst	 bmhh3, [sp, #-0x900]
     19c:      	vst	 bmhl1, [sp, #-0xb40]
     1a0:      	vst	 bmhh1, [sp, #-0xb00]
     1a4:      	vlda	 bmll3, [sp, #-0xdc0];		vst	 bmll3, [sp, #-0xac0]
     1aa:      	vst	 bmll3, [sp, #-0xa80];		vbcst.32	 x4, r2
     1b2:      	vst	 bmll1, [sp, #-0xec0];		vmov	bmlh3, bmll3
     1ba:      	vst	 bmll1, [sp, #-0xe80];		vbcst.16	 x6, r2
     1c2:      	vst	 bmll3, [sp, #-0xcc0];		vmov	cmh3, cml3
     1ca:      	vst	 bmll3, [sp, #-0xc80];		mov	p2, sp
     1d2:      	vst	 bmhl3, [sp, #-0xc40];		mov	s0, r2
     1da:      	vst	 bmhh3, [sp, #-0xc00];		vmov	bmlh3, bmll3
     1e2:      	vst	 bmll3, [sp, #-0xdc0];		vconv.fp32.bf16	cml3, x7
     1ea:      	vst	 bmll3, [sp, #-0xd80];		vmov	cmh3, cml3
     1f2:      	vst	 bmll3, [sp, #-0x7c0];		vmov	bmlh1, bmll1
     1fa:      	vlda	 bmhl3, [sp, #-0xe40];		vst	 bmlh3, [sp, #-0x780];		movx	r4, #0x33c;		mov	s2, #0x17
     206:      	vlda	 bmhh3, [sp, #-0xe00];		vst	 bmhl3, [sp, #-0xd40];		movx	r5, #0x3c;		vmov	cmh1, cml1
     212:      	vlda	 bmll3, [sp, #-0xec0];		vst	 bmhh3, [sp, #-0xd00];		movx	r7, #0x20;		mov	s3, r2
     21e:      	vlda	 bmlh3, [sp, #-0xec0];		vst	 bmhl1, [sp, #-0xe40];		movx	r6, #0x10;		vconv.fp32.bf16	cml3, x5
     22a:      	mova	r3, #0x7f;		vst	 bmhh1, [sp, #-0xe00];		movx	r28, #0x8;		mov	s1, r6
     236:      	mova	r18, #0x3;		vst	 bmll3, [sp, #-0x740];		movx	r17, #0x2;		vbcst.32	 x1, r3
     242:      	mova	r19, #0xb;		vst	 bmlh3, [sp, #-0x700];		movxm	r3, #-0xeb60d36
     24e:      	mova	r21, #0x5;		vst	 bmhl3, [sp, #-0x6c0];		movx	r20, #0x9;		vbcst.32	 x8, r3
     25a:      	mova	r22, #0x40;		vst	 bmhh3, [sp, #-0x680];		or	 r3, r2, r2;		vmov	lfh1, x8
     266:      	mova	r24, #0x0;		vst	 bmll3, [sp, #-0x340];		movx	r23, #0x1;		vbcst.64	 x8, r3:r2
     272:      	mova	r3, #0x7;		paddb	 [p2], #-0xc0;		movx	r1, #0x4;		vmov	lfl0, x8;		vst	 bmlh3, [sp, #-0x300]

00000280 <.LBB3_2>:
     280:      	nopa	;		nopb	;		lshl	 r25, r24, r23
     288:      	movs	p3, p0;		lshl	 r26, r2, r23;		mov	m0, r25
     292:      	padda	 [p3], m0;		mov	dj0, r26
     298:      	vldb	 wl4, [p3, dj0]
		...
     2a8:      	vmax_lt.bf16	 x8, r16, x4, x0
     2ac:      	add	r26, r2, #0x10
     2b0:      	lshl	 r16, r26, r23
     2b4:      	mov	dj0, r16
     2b8:      	vldb	 wl4, [p3, dj0]
		...
     2c8:      	vmax_lt.bf16	 x8, r16, x4, x0;		vmul.f	dm0, x8, x2, r4
     2d0:      	nop	
     2d2:      	add.nc	lc, r1, #-0x2;		vmul.f	dm4, x8, x2, r4
     2da:      	movxm	ls, #0x0
     2e0:      	nopa	;		nopb	;		nops	;		movxm	le, #0x0;		nopv	
     2f0:      	nopa	;		nopb	;		nops	;		nopx	;		vmov	bmll2, lfh1;		nopv	
     300:      	nopa	;		nopb	;		nops	;		nopx	;		vmov	x7, bmll0;		nopv	
     310:      	nopx	;		vmov	x8, lfh1;		vsub.f	dm2, dm2, dm0, r5
     31a:      	add	r26, r26, #0x10;		vmov	x3, bmll4

00000320 <.LBB3_3>:
     320:      	nopa	;		nopb	;		lshl	 r16, r26, r23;		nopm	
     32c:      	mov	dj0, r16
     330:      	vldb	 wl9, [p3, dj0]
     334:      	nop	
     336:      	vconv.bf16.fp32	 wl4, bmll2
     33a:      	nop	
     33c:      	vlt.bf16	 r16, x4, x6
     340:      	nop	
     342:      	vsel.32	 x8, x8, x7, r16
     346:      	vmax_lt.bf16	 x7, r16, x9, x0
     34a:      	nop	
     34c:      	vmul.f	dm4, x7, x2, r4
     350:      	nop	
     352:      	vsub.f	dm2, dm0, dm4, r5
     356:      	vmov	bmll0, x8
     35a:      	nop	
     35c:      	vmov	x7, x3

00000360 <.L_LEnd5>:
     360:      	nopa	;		nopb	;		nops	;		add	r26, r26, #0x10;		vmov	x3, bmll4;		nopv	
     370:      	nop	
     372:      	vconv.bf16.fp32	 wl4, bmll2
     376:      	nop	
     378:      	vlt.bf16	 r16, x4, x6
     37c:      	nop	
     37e:      	vsel.32	 x8, x8, x7, r16;		vsub.f	dm0, dm0, dm4, r5
     386:      	vmov	bmll0, x8
		...
     392:      	vconv.bf16.fp32	 wl4, bmll0
     396:      	nop	
     398:      	vlt.bf16	 r16, x4, x6
     39c:      	nop	
     39e:      	vsel.32	 x8, x8, x3, r16
     3a2:      	vshift	x3, x8, x0, r7
     3a6:      	vmov	bmll0, x8;		vsub.f	dm0, dm0, dm2, r5
     3ae:      	vmov	bmll2, x3
		...
     3ba:      	vconv.bf16.fp32	 wl4, bmll0
     3be:      	nop	
     3c0:      	vlt.bf16	 r26, x4, x6
     3c4:      	nop	
     3c6:      	vsel.32	 x8, x8, x3, r26
     3ca:      	vshift	x3, x8, x0, r6
     3ce:      	vmov	bmll0, x8;		vsub.f	dm0, dm0, dm2, r5
     3d6:      	vmov	bmll2, x3
		...
     3e2:      	vconv.bf16.fp32	 wl4, bmll0
     3e6:      	nop	
     3e8:      	vlt.bf16	 r26, x4, x6
     3ec:      	nop	
     3ee:      	vsel.32	 x8, x8, x3, r26
     3f2:      	vshift	x3, x8, x0, r28
     3f6:      	vmov	bmll0, x8;		vsub.f	dm0, dm0, dm2, r5
     3fe:      	vmov	bmll2, x3
		...
     40a:      	vconv.bf16.fp32	 wl4, bmll0
     40e:      	nop	
     410:      	vlt.bf16	 r26, x4, x6
     414:      	nop	
     416:      	vsel.32	 x8, x8, x3, r26
     41a:      	vshift	x3, x8, x0, r1
     41e:      	vmov	bmll0, x8;		vsub.f	dm0, dm0, dm2, r5
     426:      	vmov	bmll2, x3
     42a:      	nop	
     42c:      	add.nc	lc, r1, #0x0
     430:      	movxm	ls, #0x0
     436:      	movxm	le, #0x0
     43c:      	vlda	 bmlh0, [sp, #-0x780];		vconv.bf16.fp32	 wl4, bmll0;		vmov	lfh0, x1
     446:      	vmov	bmhl3, x10
     44a:      	vlda	 bmll0, [sp, #-0x7c0];		vlt.bf16	 r26, x4, x6
     450:      	vmov	bmll1, lfl0
     454:      	vsel.32	 x8, x8, x3, r26
     458:      	vextbcst.32	 x8, x8, #0x0
     45c:      	vst	 bmlh0, [sp, #-0x280];		vmov	bmll0, x8
     464:      	vst	 bmhl0, [sp, #-0x240];		vmov	x1, x10
     46c:      	vst	 bmll0, [sp, #-0x2c0];		vmov	lfl1, lfh0
     474:      	mova	r26, #0x0;		vst	 bmhh0, [sp, #-0x200];		movx	r27, #0x0;		vmov	cml3, cml0

00000480 <.LBB3_5>:
     480:      	lshl	 r16, r26, r23
     484:      	movs	dj0, r16
     488:      	vldb	 wl10, [p3, dj0]
		...
     498:      	vlda	 bmll1, [sp, #-0x2c0];		vmax_lt.bf16	 x3, r16, x10, x0
     49e:      	vlda	 bmlh1, [sp, #-0x280]
     4a2:      	vlda	 bmhl1, [sp, #-0x240];		vmul.f	dm0, x3, x2, r4
     4aa:      	nop	
     4ac:      	vlda	 bmhh1, [sp, #-0x200]
     4b0:      	nop	
     4b2:      	vlda	 bmll1, [sp, #-0x9c0];		vst	 bmll1, [sp, #-0x1c0]
     4b8:      	vlda	 bmlh1, [sp, #-0x980];		vst	 bmlh1, [sp, #-0x180]
     4be:      	vlda	 bmhl1, [sp, #-0x940];		vst	 bmhl1, [sp, #-0x140];		vmov	bmlh0, bmll0
     4c8:      	vlda	 bmhh1, [sp, #-0x900];		vsub.f	dm4, dm0, dm1, r5
     4d0:      	vst	 bmhh1, [sp, #-0x100];		vmov	cmh0, cml0
     4d8:      	nop	
     4da:      	nop	
     4dc:      	vsub.f	dm1, dm4, dm1, r5
		...
     4e8:      	vlda	 bmlh1, [sp, #-0xa00]
     4ec:      	vconv.bf16.fp32	 wl4, bmll1
     4f0:      	vlda	 bmll1, [sp, #-0xa40]
     4f4:      	vlt.bf16	 r16, x4, x6
     4f8:      	vmov	x5, bmll4
     4fc:      	vsel.32	 x7, x5, x11, r16
     500:      	vmov	bmll0, x7
     504:      	nop	
     506:      	vconv.bf16.fp32	 x9, cml0
     50a:      	vconv.bf16.fp32	 x8, cml1
     50e:      	vmsc.f	dm2, dm0, x9, x2, r5
     512:      	vmsc.f	dm4, dm4, x8, x2, r5
     516:      	vmov	cml4, cml1
     51a:      	nop	
     51c:      	nop	
     51e:      	nop	
     520:      	vconv.bf16.fp32	 x10, cml2
     524:      	vconv.bf16.fp32	 x3, cml4
     528:      	vmsc.f	dm1, dm2, x10, x2, r5
     52c:      	vmsc.f	dm0, dm4, x3, x2, r5
		...
     538:      	vconv.bf16.fp32	 x5, cml1
     53c:      	vconv.bf16.fp32	 x11, cml0
     540:      	vmov	bmhh3, x11
     544:      	vmul.f	dm2, x5, x11, r5
     548:      	vmul.f	dm4, x5, x3, r5
     54c:      	nop	
     54e:      	vmul.f	dm1, x10, x11, r5
     552:      	vadd.f	dm2, dm2, dm4, r5
     556:      	nop	
     558:      	vmul.f	dm0, x9, x11, r5
     55c:      	vadd.f	dm2, dm2, dm1, r5
     560:      	nop	
     562:      	vmul.f	dm4, x10, x3, r5
     566:      	vadd.f	dm2, dm2, dm0, r5
     56a:      	nop	
     56c:      	vmul.f	dm1, x8, x5, r5
     570:      	vadd.f	dm2, dm2, dm4, r5
     574:      	nop	
     576:      	vmul.f	dm0, x10, x8, r5
     57a:      	vadd.f	dm2, dm2, dm1, r5
     57e:      	nop	
     580:      	vmul.f	dm4, x9, x3, r5
     584:      	vadd.f	dm2, dm2, dm0, r5
     588:      	nop	
     58a:      	vmul.f	dm1, x9, x8, r5
     58e:      	vlda	 bmll1, [sp, #-0x8c0];		vadd.f	dm2, dm2, dm4, r5
     596:      	vlda	 bmlh1, [sp, #-0x880]
     59a:      	vlda	 bmhl1, [sp, #-0x840]
     59e:      	vlda	 bmhh1, [sp, #-0x800];		vadd.f	dm2, dm2, dm1, r5
     5a6:      	nop	
     5a8:      	nop	
     5aa:      	nop	
     5ac:      	vsub.f	dm1, dm2, dm1, r5
		...
     5b8:      	nop	
     5ba:      	vconv.bf16.fp32	 wl4, bmll1
     5be:      	nop	
     5c0:      	vlt.bf16	 r16, x4, x6
     5c4:      	vmov	x7, bmll2
     5c8:      	vsel.32	 x7, x7, x1, r16;		vadd.f	dm4, dm0, dm3, r5
     5d0:      	vmov	bmll0, x7
     5d4:      	nop	
     5d6:      	vsub	dm4, dm4, dm3, r2
     5da:      	nop	
     5dc:      	nop	
     5de:      	mov	crsrsmode, #0x0
     5e2:      	mov	r16, crsat
     5e6:      	mov	crsat, #0x1
     5ea:      	vsrs.2x	x9, cml4, s0, srssign1
     5ee:      	nop	
     5f0:      	nop	
     5f2:      	mov	crupsmode, #0x0
     5f6:      	vups.2x	cml2, x9, s0, upssign1;		vadd	dm2, dm2, dm3, r2
     5fe:      	vlda	 bmhl4, [sp, #-0x6c0]
     602:      	vlda	 bmhh4, [sp, #-0x680];		vsub.f	dm2, dm2, dm3, r5
     60a:      	vlda	 bmll4, [sp, #-0x740]
     60e:      	vlda	 bmlh4, [sp, #-0x700]
     612:      	vsub.f	dm2, dm0, dm2, r5
     616:      	nop	
     618:      	nop	
     61a:      	vadd.f	dm2, dm2, dm4, r5
     61e:      	nop	
     620:      	nop	
     622:      	vsub	dm2, dm2, dm4, r2
     626:      	nop	
     628:      	vneg	dm1, dm2, r2
     62c:      	nop	
     62e:      	nop	
     630:      	nop	
     632:      	vsrs.2x	x8, cml2, s0, srssign0
     636:      	nop	
     638:      	vsrs.2x	x10, cml1, s0, srssign0;		mov	crupsmode, #0x1
     640:      	vups.4x	dm2, x9, s1, upssign1
     644:      	vups.4x	dm1, x8, s0, upssign0;		vadd	dm2, dm2, dm1, r17
     64c:      	nop	
     64e:      	vups.4x	dm1, x10, s0, upssign0;		vsub	dm2, dm2, dm1, r17
		...
     65e:      	mov	crsrsmode, #0x1
     662:      	vsrs.2x	x3, cml2, s0, srssign1
     666:      	nop	
     668:      	mov	crupsmode, #0x0
     66c:      	mov	crsat, r16
     670:      	vshuffle	x5, x3, x0, r18
     674:      	vshuffle	x11, x3, x0, r17
     678:      	vups.2x	cml1, x5, s0, upssign1;		vadd	dm1, dm1, dm3, r2
     680:      	vups.2x	cml2, x11, s0, upssign0;		vadd	dm2, dm2, dm4, r2
     688:      	vsub.f	dm1, dm1, dm3, r5
     68c:      	vsub.f	dm2, dm2, dm4, r5
     690:      	nop	
     692:      	nop	
     694:      	vlda	 bmll1, [sp, #-0xac0];		vadd.f	dm2, dm2, dm1, r5
     69c:      	vlda	 bmlh1, [sp, #-0xa80]
     6a0:      	nop	
     6a2:      	vsub.f	dm2, dm0, dm2, r5
		...
     6ae:      	vconv.bf16.fp32	 x7, cml1
     6b2:      	vconv.bf16.fp32	 x9, cml2
     6b6:      	vmsc.f	dm0, dm0, x7, x2, r5
     6ba:      	vmov	cml0, cml1;		vmsc.f	dm4, dm2, x9, x2, r5
		...
     6ca:      	vconv.bf16.fp32	 x8, cml0
     6ce:      	vconv.bf16.fp32	 x10, cml4
     6d2:      	vmsc.f	dm1, dm0, x8, x2, r5
     6d6:      	vmsc.f	dm2, dm4, x10, x2, r5
		...
     6e2:      	vconv.bf16.fp32	 x5, cml1
     6e6:      	vconv.bf16.fp32	 x11, cml2
     6ea:      	vmul.f	dm4, x5, x10, r5
     6ee:      	vmul.f	dm0, x5, x11, r5
     6f2:      	nop	
     6f4:      	vmul.f	dm1, x8, x11, r5
     6f8:      	vadd.f	dm0, dm0, dm4, r5
     6fc:      	nop	
     6fe:      	vmul.f	dm2, x7, x11, r5
     702:      	vadd.f	dm0, dm0, dm1, r5
     706:      	nop	
     708:      	vmul.f	dm4, x8, x10, r5
     70c:      	vadd.f	dm0, dm0, dm2, r5
     710:      	nop	
     712:      	vmul.f	dm1, x9, x5, r5
     716:      	vadd.f	dm0, dm0, dm4, r5
     71a:      	nop	
     71c:      	vmul.f	dm2, x8, x9, r5
     720:      	vadd.f	dm0, dm0, dm1, r5
     724:      	nop	
     726:      	vmul.f	dm4, x7, x10, r5
     72a:      	vadd.f	dm0, dm0, dm2, r5
     72e:      	nop	
     730:      	vmul.f	dm1, x7, x9, r5
     734:      	vlda	 bmll1, [sp, #-0xbc0];		vadd.f	dm0, dm0, dm4, r5
     73c:      	vlda	 bmlh1, [sp, #-0xb80]
     740:      	vlda	 bmhl1, [sp, #-0xb40]
     744:      	vlda	 bmhh1, [sp, #-0xb00];		vadd.f	dm0, dm0, dm1, r5
     74c:      	nop	
     74e:      	nop	
     750:      	nop	
     752:      	vadd.f	dm0, dm0, dm1, r5
		...
     75e:      	nop	
     760:      	vconv.bf16.fp32	 x7, cml0
     764:      	nop	
     766:      	vmsc.f	dm2, dm0, x7, x2, r5
		...
     772:      	nop	
     774:      	vconv.bf16.fp32	 x8, cml2
     778:      	nop	
     77a:      	vmsc.f	dm4, dm2, x8, x2, r5
		...
     786:      	nop	
     788:      	vconv.bf16.fp32	 x5, cml4
     78c:      	nop	
     78e:      	vmul.f	dm1, x5, x11, r5
     792:      	vmul.f	dm0, x5, x10, r5
     796:      	nop	
     798:      	vmul.f	dm2, x8, x11, r5
     79c:      	vadd.f	dm1, dm1, dm0, r5
     7a0:      	nop	
     7a2:      	vmul.f	dm4, x7, x11, r5
     7a6:      	vadd.f	dm1, dm1, dm2, r5
     7aa:      	nop	
     7ac:      	vmul.f	dm0, x8, x10, r5
     7b0:      	vadd.f	dm1, dm1, dm4, r5
     7b4:      	nop	
     7b6:      	vmul.f	dm2, x9, x5, r5
     7ba:      	vadd.f	dm1, dm1, dm0, r5
     7be:      	nop	
     7c0:      	vmul.f	dm4, x8, x9, r5
     7c4:      	vadd.f	dm1, dm1, dm2, r5
     7c8:      	nop	
     7ca:      	vmul.f	dm0, x7, x10, r5
     7ce:      	vadd.f	dm1, dm1, dm4, r5
     7d2:      	nop	
     7d4:      	vmul.f	dm2, x7, x9, r5
     7d8:      	vlda	 bmll1, [sp, #-0x640];		vadd.f	dm1, dm1, dm0, r5
     7e0:      	vlda	 bmlh1, [sp, #-0x600]
     7e4:      	vlda	 bmhl1, [sp, #-0x5c0]
     7e8:      	vlda	 bmhh1, [sp, #-0x580];		vadd.f	dm4, dm1, dm2, r5
     7f0:      	nop	
     7f2:      	nop	
     7f4:      	nop	
     7f6:      	vadd.f	dm4, dm4, dm1, r5
		...
     802:      	nop	
     804:      	vconv.bf16.fp32	 x7, cml4
     808:      	nop	
     80a:      	vmsc.f	dm0, dm4, x7, x2, r5
		...
     816:      	nop	
     818:      	vconv.bf16.fp32	 x8, cml0
     81c:      	nop	
     81e:      	vmsc.f	dm1, dm0, x8, x2, r5
		...
     82a:      	nop	
     82c:      	vconv.bf16.fp32	 x5, cml1
     830:      	nop	
     832:      	vmul.f	dm2, x5, x11, r5
     836:      	vmul.f	dm4, x5, x10, r5
     83a:      	nop	
     83c:      	vmul.f	dm0, x8, x11, r5
     840:      	vadd.f	dm2, dm2, dm4, r5
     844:      	nop	
     846:      	vmul.f	dm1, x7, x11, r5
     84a:      	vadd.f	dm2, dm2, dm0, r5
     84e:      	nop	
     850:      	vmul.f	dm4, x8, x10, r5
     854:      	vadd.f	dm2, dm2, dm1, r5
     858:      	nop	
     85a:      	vmul.f	dm0, x9, x5, r5
     85e:      	vadd.f	dm2, dm2, dm4, r5
     862:      	nop	
     864:      	vmul.f	dm1, x8, x9, r5
     868:      	vadd.f	dm2, dm2, dm0, r5
     86c:      	nop	
     86e:      	vmul.f	dm4, x7, x10, r5
     872:      	vadd.f	dm2, dm2, dm1, r5
     876:      	nop	
     878:      	vlda	 bmll1, [sp, #-0x540];		vmul.f	dm0, x7, x9, r5
     880:      	vlda	 bmlh1, [sp, #-0x500];		vadd.f	dm2, dm2, dm4, r5
     888:      	vlda	 bmhl1, [sp, #-0x4c0]
     88c:      	vlda	 bmhh1, [sp, #-0x480]
     890:      	vadd.f	dm2, dm2, dm0, r5
     894:      	nop	
     896:      	nop	
     898:      	vadd.f	dm2, dm2, dm1, r5
		...
     8a4:      	nop	
     8a6:      	vconv.bf16.fp32	 x7, cml2
     8aa:      	nop	
     8ac:      	vmsc.f	dm1, dm2, x7, x2, r5
		...
     8b8:      	nop	
     8ba:      	vconv.bf16.fp32	 x8, cml1
     8be:      	nop	
     8c0:      	vmsc.f	dm4, dm1, x8, x2, r5
		...
     8cc:      	nop	
     8ce:      	vconv.bf16.fp32	 x5, cml4
     8d2:      	nop	
     8d4:      	vmul.f	dm0, x5, x11, r5
     8d8:      	vmul.f	dm2, x5, x10, r5
     8dc:      	nop	
     8de:      	vmul.f	dm1, x8, x11, r5
     8e2:      	vadd.f	dm0, dm0, dm2, r5
     8e6:      	nop	
     8e8:      	vmul.f	dm4, x7, x11, r5
     8ec:      	vadd.f	dm0, dm0, dm1, r5
     8f0:      	nop	
     8f2:      	vmul.f	dm2, x8, x10, r5
     8f6:      	vadd.f	dm0, dm0, dm4, r5
     8fa:      	nop	
     8fc:      	vmul.f	dm1, x9, x5, r5
     900:      	vadd.f	dm0, dm0, dm2, r5
     904:      	nop	
     906:      	vmul.f	dm4, x8, x9, r5
     90a:      	vadd.f	dm0, dm0, dm1, r5
     90e:      	nop	
     910:      	vmul.f	dm2, x7, x10, r5
     914:      	vadd.f	dm0, dm0, dm4, r5
     918:      	nop	
     91a:      	vmul.f	dm1, x7, x9, r5
     91e:      	vlda	 bmll1, [sp, #-0x440];		vadd.f	dm0, dm0, dm2, r5
     926:      	vlda	 bmlh1, [sp, #-0x400]
     92a:      	vlda	 bmhl1, [sp, #-0x3c0]
     92e:      	vlda	 bmhh1, [sp, #-0x380];		vadd.f	dm0, dm0, dm1, r5
     936:      	nop	
     938:      	nop	
     93a:      	nop	
     93c:      	vadd.f	dm0, dm0, dm1, r5
		...
     948:      	nop	
     94a:      	vconv.bf16.fp32	 x7, cml0
     94e:      	nop	
     950:      	vmsc.f	dm4, dm0, x7, x2, r5
		...
     95c:      	nop	
     95e:      	vconv.bf16.fp32	 x8, cml4
     962:      	nop	
     964:      	vmsc.f	dm2, dm4, x8, x2, r5
		...
     970:      	nop	
     972:      	vconv.bf16.fp32	 x5, cml2
     976:      	nop	
     978:      	vmul.f	dm1, x5, x11, r5
     97c:      	vmul.f	dm0, x5, x10, r5
     980:      	nop	
     982:      	vmul.f	dm4, x8, x11, r5
     986:      	vadd.f	dm1, dm1, dm0, r5
     98a:      	nop	
     98c:      	vmul.f	dm2, x7, x11, r5
     990:      	vadd.f	dm1, dm1, dm4, r5
     994:      	nop	
     996:      	vmul.f	dm0, x8, x10, r5
     99a:      	vadd.f	dm1, dm1, dm2, r5
     99e:      	nop	
     9a0:      	vmul.f	dm4, x9, x5, r5
     9a4:      	vadd.f	dm1, dm1, dm0, r5
     9a8:      	nop	
     9aa:      	vmul.f	dm2, x8, x9, r5
     9ae:      	vadd.f	dm1, dm1, dm4, r5
     9b2:      	nop	
     9b4:      	vmul.f	dm0, x7, x10, r5
     9b8:      	vadd.f	dm1, dm1, dm2, r5
     9bc:      	nop	
     9be:      	vmul.f	dm4, x7, x9, r5
     9c2:      	vlda	 bmll1, [sp, #-0xcc0];		vadd.f	dm1, dm1, dm0, r5
     9ca:      	vlda	 bmlh1, [sp, #-0xc80]
     9ce:      	vlda	 bmhl1, [sp, #-0xc40]
     9d2:      	vlda	 bmhh1, [sp, #-0xc00];		vadd.f	dm2, dm1, dm4, r5
     9da:      	nop	
     9dc:      	nop	
     9de:      	nop	
     9e0:      	vadd.f	dm2, dm2, dm1, r5
		...
     9ec:      	nop	
     9ee:      	vconv.bf16.fp32	 x7, cml2
     9f2:      	nop	
     9f4:      	vmsc.f	dm0, dm2, x7, x2, r5
		...
     a00:      	nop	
     a02:      	vconv.bf16.fp32	 x8, cml0
     a06:      	nop	
     a08:      	vmsc.f	dm1, dm0, x8, x2, r5
		...
     a14:      	nop	
     a16:      	vconv.bf16.fp32	 x5, cml1
     a1a:      	nop	
     a1c:      	vmul.f	dm4, x5, x11, r5
     a20:      	vmul.f	dm2, x5, x10, r5
     a24:      	nop	
     a26:      	vmul.f	dm0, x8, x11, r5
     a2a:      	vadd.f	dm4, dm4, dm2, r5
     a2e:      	nop	
     a30:      	vmul.f	dm1, x7, x11, r5
     a34:      	vadd.f	dm4, dm4, dm0, r5
     a38:      	nop	
     a3a:      	vmul.f	dm2, x8, x10, r5
     a3e:      	vadd.f	dm4, dm4, dm1, r5
     a42:      	nop	
     a44:      	vmul.f	dm0, x9, x5, r5
     a48:      	vadd.f	dm4, dm4, dm2, r5
     a4c:      	nop	
     a4e:      	vmul.f	dm1, x8, x9, r5
     a52:      	vadd.f	dm4, dm4, dm0, r5
     a56:      	nop	
     a58:      	vmul.f	dm2, x7, x10, r5
     a5c:      	vadd.f	dm4, dm4, dm1, r5
     a60:      	nop	
     a62:      	vlda	 bmll1, [sp, #-0xdc0];		vmul.f	dm0, x7, x9, r5
     a6a:      	vlda	 bmlh1, [sp, #-0xd80];		vadd.f	dm4, dm4, dm2, r5
     a72:      	vlda	 bmhl1, [sp, #-0xd40]
     a76:      	vlda	 bmhh1, [sp, #-0xd00]
     a7a:      	vadd.f	dm4, dm4, dm0, r5
     a7e:      	nop	
     a80:      	nop	
     a82:      	vadd.f	dm4, dm4, dm1, r5
		...
     a8e:      	nop	
     a90:      	vconv.bf16.fp32	 x7, cml4
     a94:      	nop	
     a96:      	vmsc.f	dm1, dm4, x7, x2, r5
		...
     aa2:      	nop	
     aa4:      	vconv.bf16.fp32	 x8, cml1
     aa8:      	nop	
     aaa:      	vmsc.f	dm2, dm1, x8, x2, r5
		...
     ab6:      	nop	
     ab8:      	vconv.bf16.fp32	 x5, cml2
     abc:      	nop	
     abe:      	vmul.f	dm0, x5, x11, r5
     ac2:      	vmul.f	dm4, x5, x10, r5
     ac6:      	nop	
     ac8:      	vmul.f	dm1, x8, x11, r5
     acc:      	vadd.f	dm0, dm0, dm4, r5
     ad0:      	nop	
     ad2:      	vmul.f	dm2, x7, x11, r5
     ad6:      	vadd.f	dm0, dm0, dm1, r5
     ada:      	nop	
     adc:      	vmul.f	dm4, x8, x10, r5
     ae0:      	vadd.f	dm0, dm0, dm2, r5
     ae4:      	nop	
     ae6:      	vmul.f	dm1, x9, x5, r5
     aea:      	vadd.f	dm0, dm0, dm4, r5
     aee:      	nop	
     af0:      	vmul.f	dm2, x8, x9, r5
     af4:      	vadd.f	dm0, dm0, dm1, r5
     af8:      	nop	
     afa:      	vmul.f	dm4, x7, x10, r5
     afe:      	vadd.f	dm0, dm0, dm2, r5
     b02:      	nop	
     b04:      	vlda	 bmll2, [sp, #-0xec0];		vmov	x8, lfl1;		vmul.f	dm1, x7, x9, r5
     b0e:      	vlda	 bmlh2, [sp, #-0xe80];		vadd.32	 x8, x3, x8;		vadd.f	dm0, dm0, dm4, r5
     b18:      	vlda	 bmhl2, [sp, #-0xe40];		mov	crupsmode, #0x1
     b1e:      	vlda	 bmhh2, [sp, #-0xe00];		vups.2x	cmh4, x8, s2, upssign1
     b24:      	vadd.f	dm0, dm0, dm1, r5
     b28:      	nop	
     b2a:      	vsrs.2x	x3, cmh4, s3, srssign1
     b2e:      	vadd.f	dm0, dm0, dm2, r5
     b32:      	nop	
     b34:      	nop	
     b36:      	vmov	bmll2, x3
     b3a:      	nop	
     b3c:      	vconv.bf16.fp32	 x10, cml2
     b40:      	vconv.bf16.fp32	 x9, cml0
     b44:      	vmsc.f	dm4, dm2, x10, x2, r5
     b48:      	vmsc.f	dm1, dm0, x9, x2, r5
		...
     b54:      	vconv.bf16.fp32	 x5, cml4
     b58:      	vconv.bf16.fp32	 x7, cml1
     b5c:      	vmsc.f	dm2, dm4, x5, x2, r5
     b60:      	vmsc.f	dm0, dm1, x7, x2, r5
		...
     b6c:      	vconv.bf16.fp32	 x3, cml2
     b70:      	vconv.bf16.fp32	 x8, cml0
     b74:      	nop	
     b76:      	vmul.f	dm1, x8, x3, r5
     b7a:      	vmul.f	dm4, x8, x5, r5
     b7e:      	nop	
     b80:      	vmul.f	dm0, x7, x3, r5
     b84:      	vadd.f	dm1, dm1, dm4, r5
     b88:      	nop	
     b8a:      	vmul.f	dm2, x9, x3, r5
     b8e:      	vadd.f	dm1, dm1, dm0, r5
     b92:      	nop	
     b94:      	vmul.f	dm4, x7, x5, r5
     b98:      	vadd.f	dm1, dm1, dm2, r5
     b9c:      	nop	
     b9e:      	vmul.f	dm0, x10, x8, r5
     ba2:      	vadd.f	dm1, dm1, dm4, r5
     ba6:      	nop	
     ba8:      	vmul.f	dm2, x7, x10, r5
     bac:      	vadd.f	dm1, dm1, dm0, r5
     bb0:      	nop	
     bb2:      	vmul.f	dm4, x9, x5, r5
     bb6:      	vadd.f	dm1, dm1, dm2, r5
     bba:      	nop	
     bbc:      	vmul.f	dm0, x9, x10, r5
     bc0:      	vadd.f	dm1, dm1, dm4, r5
     bc4:      	nop	
     bc6:      	vlda	 bmll1, [sp, #-0x340]
     bca:      	vlda	 bmlh1, [sp, #-0x300];		vadd.f	dm2, dm1, dm0, r5
		...
     bda:      	nop	
     bdc:      	vconv.bf16.fp32	 x9, cml2
     be0:      	vconv.bf16.fp32	 x10, cml1
     be4:      	vmsc.f	dm4, dm2, x9, x2, r5
     be8:      	vmsc.f	dm1, dm1, x10, x2, r5
		...
     bf4:      	vconv.bf16.fp32	 x7, cml4
     bf8:      	vconv.bf16.fp32	 x5, cml1
     bfc:      	vmsc.f	dm0, dm4, x7, x2, r5
     c00:      	vmsc.f	dm2, dm1, x5, x2, r5
		...
     c0c:      	vconv.bf16.fp32	 x8, cml0
     c10:      	vconv.bf16.fp32	 x3, cml2
     c14:      	vmul.f	dm1, x8, x5, r5
     c18:      	vmul.f	dm4, x8, x3, r5
     c1c:      	nop	
     c1e:      	vmul.f	dm0, x7, x3, r5
     c22:      	vadd.f	dm4, dm4, dm1, r5
     c26:      	nop	
     c28:      	vmul.f	dm2, x9, x3, r5
     c2c:      	vadd.f	dm4, dm4, dm0, r5
     c30:      	nop	
     c32:      	vmul.f	dm1, x7, x5, r5
     c36:      	vadd.f	dm4, dm4, dm2, r5
     c3a:      	nop	
     c3c:      	vmul.f	dm0, x10, x8, r5
     c40:      	vadd.f	dm4, dm4, dm1, r5
     c44:      	nop	
     c46:      	vmul.f	dm2, x7, x10, r5
     c4a:      	vadd.f	dm4, dm4, dm0, r5
     c4e:      	nop	
     c50:      	vmul.f	dm1, x9, x5, r5
     c54:      	vadd.f	dm4, dm4, dm2, r5
     c58:      	nop	
     c5a:      	vmul.f	dm0, x9, x10, r5
     c5e:      	vadd.f	dm4, dm4, dm1, r5
     c62:      	nop	
     c64:      	nop	
     c66:      	vadd.f	dm4, dm4, dm0, r5
		...
     c72:      	nop	
     c74:      	vconv.bf16.fp32	 wl4, bmll4
     c78:      	vlda	 bmll1, [sp, #-0x1c0]
     c7c:      	vlda	 bmlh1, [sp, #-0x180];		vmul.f	dm0, x4, x2, r4
     c84:      	nop	
     c86:      	vlda	 bmhl1, [sp, #-0x140]
     c8a:      	nop	
     c8c:      	vlda	 bmhh1, [sp, #-0x100];		lshl	 r16, r27, r21
     c92:      	mov	dj0, r16
     c96:      	vmov	bmlh0, bmll0
     c9a:      	add	r26, r26, #0x10;		vmov	x11, bmhh3

00000ca0 <.L_LEnd4>:
     ca0:      	nopa	;		nopb	;		vst	 wl4, [p2, dj0];		add	r27, r27, #0x1;		vmov	cmh0, cml0;		vadd.f	dm1, dm1, dm0, r5
     cb0:      	nopa	;		nopxm	
		...
     cc2:      	vmov	x8, bmll1;		vadd.f	dm0, dm1, dm0, r5
     cca:      	vshuffle	bmll0, x8, x8, r19
		...
     cd6:      	vmov	x8, bmll0;		vadd.f	dm0, dm0, dm2, r5
     cde:      	vshuffle	bmll2, x8, x8, r20
		...
     cea:      	vmov	x8, bmll0;		vadd.f	dm0, dm0, dm2, r5
     cf2:      	vshuffle	bmll2, x8, x8, r3
		...
     cfe:      	vmov	x8, bmll0;		vadd.f	dm0, dm0, dm2, r5
     d06:      	vshuffle	bmll2, x8, x8, r21
		...
     d12:      	vmov	x8, bmll0
     d16:      	vmov	x9, bmlh0
     d1a:      	vshuffle	x8, x8, x9, r1
     d1e:      	vextract.32	 r16, x8, #0x0, vaddsign1
     d22:      	nop	
     d24:      	inv	 r16, r16
     d28:      	add.nc	lc, r1, #0x0
     d2c:      	movxm	ls, #0x0
     d32:      	movs	m0, r25;		movxm	le, #0x0
     d3c:      	mova	r25, #0x0;		movs	p3, p1;		vbcst.32	 x8, r16
     d46:      	padda	 [p3], m0;		movx	r26, #0x0;		vmov	bmll4, x8

00000d50 <.LBB3_7>:
     d50:      	nopa	;		nopb	;		lshl	 r16, r26, r21;		nopm	
     d5c:      	movs	dj0, r16
     d60:      	vldb	 wl9, [p2, dj0]
		...
     d70:      	vmul.f	dm2, x9, x2, r4
		...
     d7c:      	nop	
     d7e:      	vconv.bf16.fp32	 x8, cml2
     d82:      	vconv.bf16.fp32	 x10, cml4
     d86:      	vmsc.f	dm0, dm2, x8, x2, r5
     d8a:      	vmsc.f	dm1, dm3, x10, x2, r5
     d8e:      	vmov	cml3, cml4
     d92:      	nop	
     d94:      	nop	
     d96:      	nop	
     d98:      	vconv.bf16.fp32	 x1, cml0
     d9c:      	vconv.bf16.fp32	 x3, cml1
     da0:      	vmsc.f	dm2, dm0, x1, x2, r5
     da4:      	vmsc.f	dm0, dm1, x3, x2, r5
		...
     db0:      	vconv.bf16.fp32	 x5, cml2
     db4:      	vconv.bf16.fp32	 x7, cml0
     db8:      	nop	
     dba:      	vmul.f	dm1, x5, x7, r5
     dbe:      	vmul.f	dm2, x5, x3, r5
     dc2:      	nop	
     dc4:      	vmul.f	dm0, x1, x7, r5
     dc8:      	vadd.f	dm1, dm1, dm2, r5
     dcc:      	nop	
     dce:      	vmul.f	dm2, x8, x7, r5
     dd2:      	vadd.f	dm1, dm1, dm0, r5
     dd6:      	nop	
     dd8:      	vmul.f	dm0, x1, x3, r5
     ddc:      	vadd.f	dm1, dm1, dm2, r5
     de0:      	nop	
     de2:      	vmul.f	dm2, x10, x5, r5
     de6:      	vadd.f	dm1, dm1, dm0, r5
     dea:      	nop	
     dec:      	vmul.f	dm0, x1, x10, r5
     df0:      	vadd.f	dm1, dm1, dm2, r5
     df4:      	nop	
     df6:      	vmul.f	dm2, x8, x3, r5
     dfa:      	vadd.f	dm1, dm1, dm0, r5
     dfe:      	nop	
     e00:      	vmul.f	dm0, x8, x10, r5
     e04:      	vadd.f	dm1, dm1, dm2, r5
     e08:      	nop	
     e0a:      	nop	
     e0c:      	vadd.f	dm1, dm1, dm0, r5
     e10:      	nop	
     e12:      	nop	
     e14:      	nop	
     e16:      	lshl	 r16, r25, r23
     e1a:      	add	r26, r26, #0x1;		mov	dj0, r16

00000e20 <.L_LEnd3>:
     e20:      	nopa	;		nopb	;		vst.conv.bf16.fp32	 bmll1, [p3, dj0];		add	r25, r25, #0x10;		nopm	;		nopv	
     e30:      	nopa	;		add	r0, r0, #-0x1;		nopm	
     e3a:      	jnz	 r0, #0x0
     e40:      	nop	
     e42:      	nop	
     e44:      	nop	
     e46:      	vmov	x10, bmhl3
     e4a:      	add	 r24, r24, r22;		vmov	x1, lfh0

00000e50 <.LBB3_9>:
     e50:      	nopa	;		nopb	;		nops	;		ret	lr;		nopm	;		nopv	
     e60:      	nop	
     e62:      	nop	
     e64:      	nop	
     e66:      	paddxm	 [sp], #-0xec0
     e6c:      	event	#1

Disassembly of section .text._Z16softmax_il4_implP8bfloat16S0_i:

00000000 <_Z16softmax_il4_implP8bfloat16S0_i>:
