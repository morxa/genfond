(define (problem tireworld-099-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 - location)
    (:init (road l0 l69) (road l1 l25) (road l1 l65) (road l10 l24) (road l11 l8) (road l12 l15) (road l13 l44) (road l14 l27) (road l15 l52) (road l16 l37) (road l17 l54) (road l17 l68) (road l18 l57) (road l18 l94) (road l19 l97) (road l2 l79) (road l20 l40) (road l21 l53) (road l22 l84) (road l23 l72) (road l24 l35) (road l25 l56) (road l26 l50) (road l27 l48) (road l28 l54) (road l29 l62) (road l30 l49) (road l31 l73) (road l32 l13) (road l33 l78) (road l34 l33) (road l35 l70) (road l36 l47) (road l37 l98) (road l38 l21) (road l38 l92) (road l39 l81) (road l4 l83) (road l40 l4) (road l41 l39) (road l42 l12) (road l43 l67) (road l44 l82) (road l44 l89) (road l45 l10) (road l46 l86) (road l47 l66) (road l48 l26) (road l49 l78) (road l5 l87) (road l50 l16) (road l51 l64) (road l52 l46) (road l53 l19) (road l54 l17) (road l55 l85) (road l56 l28) (road l57 l80) (road l58 l9) (road l59 l76) (road l6 l23) (road l60 l6) (road l60 l93) (road l61 l45) (road l62 l11) (road l63 l71) (road l64 l20) (road l64 l3) (road l65 l1) (road l65 l93) (road l66 l14) (road l67 l55) (road l68 l29) (road l69 l30) (road l7 l74) (road l70 l43) (road l71 l32) (road l72 l61) (road l73 l65) (road l74 l2) (road l75 l77) (road l75 l89) (road l76 l5) (road l77 l36) (road l78 l86) (road l79 l42) (road l8 l34) (road l80 l59) (road l81 l51) (road l82 l90) (road l83 l58) (road l84 l29) (road l85 l0) (road l86 l18) (road l87 l63) (road l88 l41) (road l88 l96) (road l89 l75) (road l9 l58) (road l9 l95) (road l90 l22) (road l91 l38) (road l92 l95) (road l93 l60) (road l94 l81) (road l95 l88) (road l96 l7) (road l97 l29) (road l98 l31) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l37) (spare-in l38) (spare-in l43) (spare-in l46) (spare-in l47) (spare-in l5) (spare-in l51) (spare-in l53) (spare-in l55) (spare-in l62) (spare-in l64) (spare-in l66) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l89) (spare-in l93) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l97) (spare-in l98) (vehicle-at l91))
    (:goal (vehicle-at l3))
)