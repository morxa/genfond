(define (problem tireworld-097-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 - location)
    (:init (road l0 l22) (road l0 l63) (road l1 l60) (road l1 l73) (road l10 l23) (road l10 l77) (road l11 l61) (road l11 l91) (road l12 l52) (road l12 l76) (road l13 l64) (road l13 l73) (road l14 l35) (road l14 l51) (road l15 l81) (road l15 l87) (road l16 l33) (road l16 l93) (road l17 l65) (road l17 l76) (road l17 l90) (road l18 l3) (road l18 l96) (road l19 l23) (road l19 l31) (road l2 l36) (road l2 l88) (road l20 l6) (road l20 l75) (road l21 l71) (road l21 l92) (road l22 l0) (road l22 l48) (road l23 l10) (road l23 l19) (road l24 l46) (road l24 l65) (road l25 l30) (road l25 l33) (road l26 l55) (road l26 l57) (road l27 l42) (road l27 l54) (road l28 l56) (road l28 l70) (road l29 l5) (road l29 l88) (road l3 l18) (road l3 l37) (road l30 l25) (road l30 l77) (road l31 l19) (road l31 l37) (road l32 l9) (road l32 l96) (road l33 l16) (road l33 l25) (road l34 l43) (road l34 l79) (road l35 l14) (road l35 l59) (road l36 l2) (road l36 l59) (road l37 l3) (road l37 l31) (road l38 l46) (road l38 l83) (road l39 l42) (road l39 l50) (road l4 l52) (road l4 l75) (road l40 l78) (road l40 l84) (road l41 l72) (road l41 l78) (road l42 l27) (road l42 l39) (road l43 l34) (road l43 l67) (road l44 l58) (road l44 l68) (road l45 l49) (road l45 l95) (road l46 l24) (road l46 l38) (road l47 l57) (road l47 l9) (road l48 l22) (road l48 l55) (road l49 l45) (road l49 l7) (road l5 l29) (road l5 l61) (road l50 l39) (road l50 l63) (road l51 l14) (road l51 l58) (road l52 l12) (road l52 l4) (road l53 l89) (road l53 l94) (road l54 l27) (road l54 l74) (road l55 l26) (road l55 l48) (road l56 l28) (road l56 l95) (road l57 l26) (road l57 l47) (road l58 l44) (road l58 l51) (road l59 l35) (road l59 l36) (road l6 l20) (road l6 l68) (road l60 l1) (road l60 l62) (road l61 l11) (road l61 l5) (road l62 l60) (road l62 l69) (road l63 l0) (road l63 l50) (road l64 l13) (road l64 l85) (road l65 l17) (road l65 l24) (road l66 l74) (road l67 l43) (road l67 l81) (road l68 l44) (road l68 l6) (road l69 l62) (road l69 l87) (road l7 l49) (road l7 l8) (road l70 l28) (road l70 l91) (road l71 l21) (road l71 l84) (road l72 l41) (road l72 l86) (road l73 l1) (road l73 l13) (road l73 l80) (road l74 l54) (road l74 l66) (road l74 l80) (road l75 l20) (road l75 l4) (road l76 l12) (road l76 l17) (road l77 l10) (road l77 l30) (road l78 l40) (road l78 l41) (road l79 l34) (road l79 l89) (road l8 l7) (road l8 l86) (road l80 l73) (road l80 l74) (road l81 l15) (road l81 l67) (road l82 l83) (road l82 l94) (road l83 l38) (road l83 l82) (road l84 l40) (road l84 l71) (road l85 l64) (road l85 l90) (road l86 l72) (road l86 l8) (road l87 l15) (road l87 l69) (road l88 l2) (road l88 l29) (road l89 l53) (road l89 l79) (road l9 l32) (road l9 l47) (road l90 l17) (road l90 l85) (road l91 l11) (road l91 l70) (road l92 l21) (road l92 l93) (road l93 l16) (road l93 l92) (road l94 l53) (road l94 l82) (road l95 l45) (road l95 l56) (road l96 l18) (road l96 l32) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l20) (spare-in l27) (spare-in l29) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l6) (spare-in l69) (spare-in l70) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l84) (spare-in l86) (spare-in l87) (spare-in l9) (spare-in l91) (spare-in l94) (spare-in l95) (vehicle-at l66))
    (:goal (vehicle-at l60))
)