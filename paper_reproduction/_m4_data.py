"""Compact embedded M4 H1 data used by Figures 10--13.

The payload contains the first 700 observations of M4 Hourly series H1 and
the corresponding archived multiplicative trend.  It is gzip-compressed and
base85-encoded so the clean release does not need the 2.2 MB full M4 CSV.
"""

from __future__ import annotations

import base64
import gzip
import json

import numpy as np


_PAYLOAD = (
    "ABzY8Wu04U0{?YeORglj4ZKUQ2M?(qCCycgSs1XlUJML)VfgMQgQ}Dms_8{_L8X)^eu6<N-+zDo`S<_)`u%V6w4dj<He`RcX`d<S"
    "XRXCPbJou?ZDtwx-XDA~81I;4BemI_It$jR`YFlAMQr7s?AMMty$}1GHfKD7PmFZT%(g;*CZ2sD8|2K`OH77Z@I67o2j8E%C8?0V"
    "Cp?97;kzVzR!YNndseWoBuI-jMX)bUC!@bIh+u`1u^P^pvCF_`gH+YE0X?|1E=d`X2hV2v-ln{3c0W5b*0GvvV<eMg;6&rWOlZC*"
    "NCnLc^eNClgA5I4ti5fiia?%;@6cFzwmkS%U~z$NJG`P}&7L3|p6WPDFFp>P;JnQ(E_I*IEh4_}fS<_WKM9`|pFZMt#Qf%3g@n-m"
    "`h>SM_gHAA!H&&61e$4hwn5IN4YyHuyU6xr_u*#arP^vm@+(z+^0r`u|2)3_*gmst2dhDcU7#hLaecxHy|~q&=jIy2YS3+!xjn>3"
    "Tn{+u8mZ8R^zF}Myq?+N$sNBA$T^_%0ZR?&WI)2<Gd;>0JRI5(_*kp?)pKnw6+V+|uWT=BZXNfZxuXesUfRI-<-JQDd}fb&$ALq8"
    "X}nk=@b-pStBCyy8!d~ZLo=?o+Fn0dUNOfT%S-=V-Z)w^WZ6;nxV>OgXuAXVq`N%u@Cv&?^R*qjyN{*$`q)Q59+l5V9-vBx#yZY5"
    "(qXCUpFRQ!5oH~b)v>$yx$tWn^^Ju(-9um_m(y+MvG0~09;a9b*uIb!(Km430X+|aY>q{+^6366<N@;dQyrT0SXemVnBXV`371q0"
    "AvaDo`1!Kbg#Me`sBVviv;(<m;EV&gV_?m(Ec?9AdOQle)K9viY)O)~bzDo*Umt6^y}3qwO}7YqbwHn^xsB?!Q9kX}UJUP!r3)w1"
    "n}b#sn`5K2<79W-9$PceRhwJG9eZ`xhVP404CIah4G-))noC<=sg{SszS8jtT{Xl=bB!U29qa0BFB4dr!P%WNcwDSpFw=oH+g=l~"
    "Hj{sc4Bev-o-;kO?5JP<1DTu0xMR`E0{PXO|Jv&FMuB*Sa#V3`c%)-|d5in7<791b?02mHv6p96w-<IPee0w<&++Wm4*wa4{~(ik"
    "HZPtF=0+R05!MMd0#5qOPo7Pihd4ejf3B~ee7F!KakPXVF4nQ)9V@_NJFM)QUuXja_Q;yZYZHDlVEuvovb@?Ov^Xl}*AtxSHO~>#"
    "efWrX$4Sq{kGxz(rpHLyIl}jH>$sL&Co*2iWz5aQFvoFqu~wjTm9aJ8Im@vqADWS8Gm7dt9V=s`H)mgcdk$<mAJ@`+zmMKt6dEbN"
    "&x&VfXJyVzinEBrpFO^w%_KjZ*)zrS=IAI1d}frxlb@X_y8BPv`Af!|ojnsZ$iG;t>tWf*QK%mN4Efyyq3x9~z~vV5c$Qz8qHcsN"
    "jyaIW`M)!NoYxt6>z=R6p_#V*$0b>;4>PN0_W63?%G910@{z?{W6qi9$;8@w3h{N?cyrLoT+gw4JJ^nYI$K-%d5(PgZ2WNqeooHg"
    "(OqAgE)IZkf&9|X%-&z*SurK`sL1;EXJ_*M-K!F>Vw`RIj-ESa*(l_w=r&psx@8?dpI5?_DRO?<ujh!`WwSDuGv_1X+MC@R&&7Fy"
    "d-aOzxpV8vp|Tl>XV_I4^_RtYUiO^mwZ^hk?HuneXGvR5#7?V6WjdX=x0f3)w(Ob0qYQpBX5aY>Yc4MB`P6G4kFUDx6i49YGoIzg"
    "wrRsUpZM~~^>|M^-r<=Om_MPn2@6hVt)uUZ@7V6u>-1`Oy!gNJ`--x;+3B-w+Ey(oJHN}3*`0SS4mNiN$vXx=9-g=V*vKEh{rdNx"
    "|NPfKe*OM8iyY-^>VWuH3ve(gkUs0*RGHF`a+4(fPt&=JnfGG}X;2;SeXJuKl?c6$6#!V?o0SQ$kz+FfiAcw088pA`y~;o-Igh;>"
    "5>7exQhC&gj)N7IS)IrZ2An(2Bf6<_nNzR=);aeK%$Yxj6h*gCtM+4317oW5_@^W7R1}^PQ&c_Ys5}b#=bZ7=BsYL-J@#no4$n_+"
    "sK|0So+wW!DZffd7&#B+b5cCHZcvVY6|tNSX)4f9B<kY%ndi|}r}Z?pbnsZM?N6O*RGQiJOA59ru-~z#ZK@yxV;R^S66?a67)cx*"
    "@r7H|WCcd@<oVNUu#sk6@&^_0bh(0&L8nRAiBs?y8_B_Z;$4{>+){DKZEau{e^N)apSDTOEpsp_NP5(dnlu{6h~Xi+3UBl(TEJ+I"
    "MIw4{CXQ&%IM1icq-Y8OqMbHY4^te}We)E)uB=>47gSOe9l}PGNS<3bNz(Cijfs&Kx3<-K`}#{JJ=#c#N68458h1DYD&cUXM@`}}"
    "d`xT6<fd7**}1Z6e};g!;Ehpjpm2s^LQ87KIcj@%6pg;qn(B+19cBz4<sLQSmu2e9^9bYAJJS^j6dM{p4x<`%(RxOh&-J=D0OO{F"
    "=G75D2CXJyI*!_l04H6dJ~1}3XfBfgD>kGFFNUO1@8E3}Z0Z!CfXiKza`e1{q4OvOJ#TYU;~YWP1%*+J2P<CBIhaH)S5WAlu4|&G"
    "-D+?{8#$FxFcRX~R~jS%l2gMM(twJ~&+Bm`1X>x)lZFI^cV`$uOU8q0)Q(Z0(@X*%Ah$1x9aw#dy{i+Jz)SPu3;_d-nnj#!!jQMh"
    "$G*_gdJ^Qk3feiJ=WWK$;vyQsl!KIklqjY$HO{)7Y+?+A6rdiTBcv)}IPqwr&olzqSO-1IW55>u$MQjn$gcT-pPE5thv$exqBTLZ"
    "zSSZ<?xmbMMVXB{#=W%KAi!2meV$!+FAoE@j!@z&E0LHpMXFUYMU{;Zlq-Ra4rdi(LB1=O!mOa8$-)#($$cbbHUjxN5n3A|TYuT0"
    "deVUC(GYg<#>#mD?Lp&~a_P7XI<7gLT`m){!Ni>7<zfT!p$-y&s{X|Y9WZmMv5`oYR5ehp{uk4njUdgjcP5M!61`rEE6kpBq(|?x"
    "5%TF0&fvRK9O_7IJQC{=2p=j68#$AWkG#sgZNuzH6&D#~ZqB(`r|a>M7GRr@(9I#3wv8{YM(k2YDE{*N`1ivFk@<H#g~kB!U~BKl"
    "3RbeI#uAToB56_v)pDX$Ho0}Z#HiUBb}h>SwX(C+M%ho4H&NVGp_<7?Vs-FsP7igu9-<EBu<c<NQy%BmZiPWZ)9E?4;trg2pmkY}"
    "$I-fs4yaWadHPj_*?oAAm@kF*vMMvuc54MhU@@8#>U2Fe`=bn}LUw6MFsmbEuWnWO^Bh{$k;pT}>bqDO^ef34&5SitzmTb~YS(Ue"
    ">Fi3sNMoHzj*U<ZSr;&#y-(J&t#j>l3g+^73E72VOiw|Squ20w#o5nb8m^xRkgCHorw-S1;gRz=_3NC((A%%nZtdI4UX+`4opt2I"
    "xu&PfaOrZ1hlARR0yy0T%Ien~4TX1*i;W;Y8lEObWi<oEzy(9XOx;F+U2;K<j*O@rjVU5G!o`<`QsmDzvfTTj!r{w^MPiH>TQIui"
    "mrVz!YZY~pG6v@gnMBA<9wH$W(eBlx1dRp8ii^xxAU|==)zukS=VO$W&9ru~&&G$+3c^W^R>1`0V=lOx58GTvsHdZK2SMVzS~)Yb"
    "l4oAbcxER2vv3m8B9Zd>>`cd}Cja01^N{292bBKMo{hvjXe05MwvJpQtcrQbIRdpw7>Uhflbiz8t}i0COyTgJkyth8gu21AUZ#l&"
    "y18j0Jye6T^#|3~<|-t!7V@#VD&CO_sidsnRp;bjiJj(xd_f50Z}(U)8Gh0~>`a4*vroR(&(vu*yp0`uq=qY~Z@A!*7@o=Bk01bP"
    ")WiQar(Yw3v$ITU`Thlwpj>kU(Zi|AR;JvrbIrLycifL|WqBkf@T<3Ne2sPRTwLtp8rJ%cIq>F&)YmscDjZ}j5{iyHa!}S}V5ppy"
    "I7xf1MX%;SOI(8JqD*BAr)m;KATIB9vr4{%GBe~Dx#XOlgQ%;nfxux^uQ?aCIC&$YNR&Z|VvA3Wi${&jEg;1NDm_Q#5K|<nSE}Nt"
    "xQ&KX6*WJvK$z)DLl+3>UQ{KG*lTo7OUN%*r{BsG@(sSLGLJ&ZC+t`?T@9Mg8wg&ZUVKukVeWzSFSU5O8XMdvY0TWL!$IAPtB;q)"
    "%I2wYw<i@Yi&N%`GUe-br9m_4Dz2!XAn3Yrd7K!r>^&kvntHMAE5Td!NJMaQ!1r>6Q~waJhXH+D1@pL?o0@&U5UiNGX1TJsT&#kN"
    "J!3hskh2$lViVb3l3{MdZ8Dn^k8JBO%bDEdnXzkB!TfSeHFkE=M&+_rQHK1fUmELI<GZ81Qe*YF3JZqbu^5MRHmzI%^rcKHER7;T"
    "Lqg1VlOt~3KD@gPjzS$Fg5L&j-EI(iSXtIbA=Y!TC%Kem$>;8xQ2g|3uW?nC*DDOfXuMV~m#NNWycs|N(96|<xH_};&eMpEhsIR5"
    "%a;}l+2V_PU3|lC|MFRO;C**5ms>!)4u!`W#SBMgQgf$^_E7Fp_CYw>GFMK@Yf$yoxf+j)_|?w+CrwLUAl7#MR4;_RAfR3s?Ul<c"
    ">yQ$57rqF^U;a>|R>E6J2_sZ$m&=Yt_??)-_jS0zi6CWd;>>qMJaW;l%&)Lme&!2}!GumTHfOtjU9fz+Dxvbx-gTzTe}U5dszka-"
    "XyP|$?4^em?@ZFwl1x{hD1B-;{qp%QI^~yDJ{{MbuIqxvu4dN-<RZ^y{=9@x4n>Wl(wLB!s{k)oc<2aUAoOpnEh3K2q583OM(uK0"
    "-R}FYIHBZek#pTqxDLyed%w8;1!1Lrxz6MZ)3&$Z1?6)ab6vfCncUXJYh@O%cB8=YyPS2SOP5viVz?Kc<V#O>R~!o$yFj`|m<3<A"
    "J9&P&R&@w1{AQM&lqbs<BU9Mchr4Apf4MKB@KTJ2!rJxLRl6+Ou8V|rfwaA|N6p;&wht(w8fEU>4Cl5+xl(L=8)em`;e8@Bk>b_j"
    "_^4ZnHsz<$Tm{zkB#Jrte$-yKio8XsDs?mFm1^fzEzI_XruVBINJX=~?+F4OJ1XDWmW5Pne_<6P!pM4z2peJ2zpmEp&q^gER{Qf;"
    "1hM_aZ^K_qMeOWp1!K1tyF)^;k%-*($B>a&+DPn};twVz&UyH$rxTv_XT8sB`Xr~A*UJv}o^x!vkEo@N;;quPl>|&^-en9gl;P^&"
    "Ft~H+)4s`B`tel49qHLg$=(>)PmSz<&3ou6Bk+&Cs%L;>uf8CBKlO}1{`Y@>O{k+^J^%m"
)


def load_m4_h1() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(observed, archived_trend)`` as independent float arrays."""

    decoded = gzip.decompress(base64.b85decode(_PAYLOAD.encode("ascii")))
    values = json.loads(decoded)
    return np.asarray(values["raw"], dtype=float), np.asarray(values["trend"], dtype=float)
