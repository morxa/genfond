(define (problem tireworld-097-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 - location)
    (:init (road l0 l59) (road l1 l18) (road l1 l25) (road l1 l27) (road l1 l42) (road l10 l84) (road l11 l85) (road l12 l65) (road l12 l74) (road l13 l47) (road l13 l89) (road l14 l55) (road l15 l80) (road l16 l60) (road l17 l9) (road l18 l1) (road l19 l40) (road l2 l63) (road l20 l18) (road l20 l39) (road l21 l66) (road l22 l61) (road l23 l77) (road l24 l16) (road l25 l64) (road l26 l34) (road l27 l1) (road l27 l19) (road l28 l15) (road l29 l75) (road l29 l81) (road l3 l4) (road l30 l50) (road l30 l84) (road l31 l5) (road l31 l72) (road l32 l1) (road l33 l1) (road l33 l2) (road l34 l26) (road l34 l39) (road l35 l28) (road l36 l8) (road l37 l57) (road l38 l73) (road l38 l91) (road l39 l20) (road l4 l90) (road l40 l52) (road l41 l48) (road l42 l1) (road l42 l56) (road l43 l95) (road l44 l32) (road l45 l43) (road l46 l51) (road l46 l70) (road l46 l91) (road l47 l13) (road l48 l41) (road l48 l54) (road l49 l83) (road l5 l31) (road l50 l30) (road l50 l62) (road l51 l46) (road l51 l7) (road l52 l5) (road l53 l82) (road l53 l87) (road l54 l11) (road l55 l44) (road l56 l42) (road l56 l96) (road l57 l86) (road l58 l36) (road l58 l82) (road l6 l3) (road l60 l69) (road l61 l0) (road l61 l95) (road l62 l26) (road l63 l2) (road l63 l84) (road l64 l94) (road l65 l12) (road l65 l45) (road l66 l41) (road l67 l23) (road l68 l21) (road l68 l88) (road l69 l6) (road l7 l49) (road l70 l46) (road l71 l22) (road l72 l24) (road l72 l31) (road l73 l35) (road l74 l12) (road l75 l14) (road l76 l47) (road l76 l87) (road l77 l37) (road l78 l80) (road l78 l93) (road l79 l17) (road l8 l36) (road l8 l69) (road l80 l78) (road l81 l29) (road l81 l89) (road l82 l33) (road l82 l53) (road l82 l58) (road l82 l9) (road l83 l49) (road l83 l58) (road l84 l30) (road l84 l71) (road l85 l1) (road l86 l88) (road l87 l76) (road l88 l68) (road l88 l86) (road l89 l81) (road l9 l82) (road l90 l10) (road l91 l38) (road l92 l67) (road l93 l74) (road l93 l78) (road l94 l70) (road l95 l61) (road l96 l56) (road l96 l79) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l28) (spare-in l29) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l70) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l80) (spare-in l82) (spare-in l83) (spare-in l85) (spare-in l86) (spare-in l88) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l93) (spare-in l94) (spare-in l95) (vehicle-at l92))
    (:goal (vehicle-at l59))
)