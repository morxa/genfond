(define (problem tireworld-099-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 - location)
    (:init (road l0 l53) (road l1 l45) (road l10 l37) (road l11 l80) (road l12 l29) (road l12 l38) (road l13 l48) (road l14 l55) (road l15 l4) (road l16 l6) (road l17 l64) (road l17 l90) (road l18 l29) (road l18 l98) (road l19 l46) (road l2 l39) (road l20 l54) (road l20 l9) (road l21 l14) (road l21 l48) (road l22 l70) (road l23 l95) (road l24 l81) (road l25 l88) (road l26 l3) (road l26 l71) (road l27 l72) (road l28 l50) (road l28 l78) (road l29 l12) (road l29 l18) (road l3 l26) (road l30 l79) (road l30 l92) (road l31 l58) (road l31 l80) (road l32 l49) (road l33 l55) (road l34 l13) (road l34 l76) (road l35 l85) (road l36 l34) (road l36 l46) (road l37 l51) (road l37 l87) (road l38 l12) (road l38 l41) (road l39 l2) (road l39 l83) (road l4 l0) (road l40 l61) (road l40 l81) (road l41 l38) (road l42 l84) (road l43 l1) (road l43 l9) (road l44 l70) (road l44 l86) (road l45 l7) (road l46 l36) (road l47 l33) (road l47 l48) (road l48 l10) (road l48 l21) (road l48 l47) (road l48 l71) (road l48 l78) (road l49 l60) (road l5 l74) (road l50 l28) (road l51 l19) (road l52 l15) (road l53 l9) (road l54 l82) (road l55 l14) (road l55 l33) (road l56 l82) (road l56 l94) (road l57 l97) (road l58 l31) (road l58 l5) (road l59 l32) (road l6 l59) (road l60 l17) (road l60 l49) (road l61 l22) (road l62 l67) (road l62 l95) (road l63 l74) (road l63 l91) (road l64 l3) (road l65 l93) (road l66 l15) (road l67 l62) (road l67 l66) (road l68 l75) (road l68 l76) (road l69 l23) (road l7 l45) (road l7 l75) (road l70 l22) (road l70 l44) (road l71 l48) (road l71 l50) (road l71 l88) (road l72 l77) (road l73 l96) (road l73 l97) (road l74 l63) (road l75 l68) (road l75 l7) (road l76 l34) (road l77 l16) (road l78 l28) (road l78 l48) (road l79 l30) (road l8 l42) (road l80 l31) (road l81 l40) (road l82 l56) (road l83 l39) (road l83 l57) (road l84 l35) (road l84 l42) (road l85 l89) (road l86 l44) (road l86 l65) (road l87 l37) (road l87 l52) (road l88 l25) (road l88 l71) (road l89 l25) (road l9 l20) (road l9 l43) (road l90 l24) (road l91 l41) (road l92 l27) (road l93 l65) (road l93 l69) (road l94 l11) (road l95 l23) (road l95 l62) (road l96 l8) (road l97 l73) (road l98 l2) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l35) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l53) (spare-in l54) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l77) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l97) (spare-in l98) (vehicle-at l79))
    (:goal (vehicle-at l14))
)