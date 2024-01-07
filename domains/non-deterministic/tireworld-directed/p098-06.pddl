(define (problem tireworld-098-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 - location)
    (:init (road l0 l15) (road l0 l23) (road l1 l67) (road l1 l74) (road l10 l39) (road l10 l4) (road l11 l38) (road l11 l52) (road l12 l57) (road l12 l58) (road l13 l35) (road l13 l47) (road l14 l47) (road l15 l0) (road l15 l68) (road l16 l53) (road l17 l43) (road l17 l86) (road l18 l87) (road l19 l21) (road l19 l33) (road l2 l39) (road l2 l71) (road l20 l5) (road l20 l90) (road l21 l19) (road l21 l88) (road l22 l28) (road l22 l75) (road l23 l0) (road l23 l44) (road l24 l39) (road l24 l54) (road l25 l5) (road l25 l70) (road l25 l72) (road l26 l65) (road l26 l94) (road l27 l6) (road l27 l97) (road l28 l22) (road l28 l79) (road l29 l81) (road l29 l96) (road l3 l86) (road l3 l93) (road l30 l66) (road l31 l6) (road l31 l76) (road l32 l80) (road l32 l83) (road l33 l19) (road l33 l42) (road l34 l36) (road l34 l73) (road l35 l13) (road l35 l76) (road l36 l34) (road l36 l92) (road l37 l45) (road l37 l91) (road l38 l11) (road l38 l39) (road l38 l55) (road l39 l2) (road l39 l38) (road l39 l59) (road l39 l69) (road l4 l10) (road l40 l46) (road l40 l87) (road l41 l72) (road l41 l75) (road l42 l33) (road l42 l73) (road l43 l18) (road l44 l23) (road l44 l74) (road l45 l37) (road l45 l90) (road l46 l40) (road l46 l56) (road l47 l13) (road l47 l14) (road l48 l82) (road l49 l68) (road l49 l78) (road l5 l20) (road l5 l25) (road l50 l88) (road l50 l93) (road l51 l60) (road l51 l85) (road l52 l11) (road l52 l9) (road l53 l16) (road l53 l80) (road l54 l24) (road l54 l77) (road l55 l38) (road l55 l71) (road l56 l46) (road l56 l62) (road l57 l12) (road l57 l82) (road l58 l14) (road l59 l39) (road l6 l27) (road l6 l31) (road l60 l51) (road l60 l67) (road l61 l7) (road l61 l85) (road l62 l56) (road l62 l70) (road l62 l78) (road l63 l48) (road l64 l39) (road l64 l89) (road l65 l26) (road l65 l69) (road l66 l16) (road l66 l30) (road l67 l1) (road l67 l60) (road l68 l15) (road l69 l39) (road l69 l65) (road l7 l61) (road l7 l95) (road l70 l25) (road l70 l62) (road l71 l2) (road l71 l55) (road l72 l41) (road l73 l34) (road l73 l42) (road l74 l1) (road l74 l44) (road l75 l22) (road l75 l41) (road l76 l31) (road l76 l35) (road l77 l54) (road l77 l95) (road l78 l49) (road l78 l62) (road l79 l28) (road l79 l4) (road l8 l84) (road l8 l97) (road l80 l32) (road l80 l53) (road l81 l29) (road l81 l38) (road l81 l83) (road l81 l92) (road l82 l57) (road l83 l81) (road l84 l8) (road l84 l9) (road l85 l51) (road l85 l61) (road l86 l17) (road l86 l3) (road l87 l40) (road l88 l50) (road l89 l64) (road l89 l96) (road l9 l52) (road l9 l84) (road l90 l20) (road l90 l45) (road l91 l37) (road l91 l63) (road l92 l36) (road l92 l81) (road l93 l3) (road l93 l50) (road l94 l26) (road l94 l30) (road l95 l7) (road l95 l77) (road l96 l89) (road l97 l8) (spare-in l10) (spare-in l12) (spare-in l15) (spare-in l20) (spare-in l23) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l49) (spare-in l55) (spare-in l58) (spare-in l64) (spare-in l67) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l82) (spare-in l88) (spare-in l90) (spare-in l92) (spare-in l94) (vehicle-at l81))
    (:goal (vehicle-at l59))
)