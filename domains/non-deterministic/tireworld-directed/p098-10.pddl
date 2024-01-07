(define (problem tireworld-098-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 - location)
    (:init (road l0 l22) (road l1 l17) (road l10 l2) (road l10 l85) (road l11 l74) (road l11 l81) (road l11 l9) (road l12 l57) (road l12 l75) (road l13 l83) (road l14 l2) (road l15 l22) (road l15 l84) (road l16 l48) (road l16 l53) (road l17 l1) (road l17 l28) (road l18 l38) (road l18 l55) (road l19 l2) (road l19 l27) (road l2 l10) (road l2 l19) (road l20 l91) (road l21 l34) (road l21 l85) (road l22 l0) (road l23 l44) (road l23 l5) (road l24 l56) (road l25 l74) (road l25 l8) (road l26 l24) (road l26 l3) (road l26 l37) (road l27 l40) (road l28 l17) (road l28 l23) (road l29 l33) (road l3 l26) (road l30 l86) (road l31 l90) (road l32 l20) (road l33 l29) (road l33 l81) (road l34 l21) (road l34 l63) (road l35 l36) (road l35 l9) (road l36 l13) (road l36 l35) (road l37 l26) (road l37 l97) (road l38 l18) (road l38 l66) (road l39 l75) (road l39 l88) (road l4 l6) (road l4 l79) (road l40 l27) (road l40 l53) (road l41 l59) (road l41 l86) (road l42 l45) (road l42 l96) (road l43 l73) (road l43 l95) (road l44 l23) (road l44 l58) (road l45 l42) (road l46 l69) (road l46 l83) (road l47 l53) (road l48 l55) (road l48 l6) (road l49 l51) (road l49 l94) (road l5 l30) (road l50 l32) (road l51 l35) (road l51 l49) (road l52 l77) (road l53 l16) (road l53 l40) (road l53 l47) (road l53 l93) (road l54 l61) (road l54 l70) (road l55 l18) (road l55 l48) (road l56 l66) (road l57 l12) (road l58 l44) (road l59 l14) (road l6 l4) (road l60 l84) (road l60 l88) (road l61 l54) (road l61 l68) (road l62 l7) (road l63 l89) (road l64 l62) (road l64 l72) (road l65 l51) (road l65 l68) (road l66 l38) (road l66 l56) (road l67 l52) (road l68 l65) (road l69 l46) (road l69 l50) (road l7 l62) (road l7 l67) (road l70 l53) (road l70 l54) (road l71 l78) (road l71 l92) (road l72 l64) (road l72 l78) (road l73 l43) (road l74 l11) (road l74 l25) (road l75 l12) (road l75 l39) (road l76 l31) (road l76 l97) (road l77 l57) (road l78 l71) (road l78 l72) (road l79 l4) (road l79 l80) (road l8 l72) (road l80 l79) (road l80 l82) (road l81 l11) (road l81 l33) (road l82 l29) (road l82 l80) (road l83 l13) (road l83 l46) (road l84 l15) (road l84 l60) (road l85 l10) (road l85 l21) (road l86 l41) (road l87 l3) (road l88 l60) (road l89 l61) (road l9 l11) (road l9 l35) (road l90 l31) (road l90 l73) (road l91 l20) (road l91 l47) (road l92 l71) (road l93 l53) (road l93 l92) (road l94 l58) (road l95 l1) (road l96 l87) (road l97 l37) (road l97 l76) (spare-in l1) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l37) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l57) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l62) (spare-in l64) (spare-in l67) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l90) (spare-in l92) (spare-in l93) (spare-in l95) (spare-in l96) (spare-in l97) (vehicle-at l45))
    (:goal (vehicle-at l0))
)