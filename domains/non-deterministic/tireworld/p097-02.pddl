(define (problem tireworld-097-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 - location)
    (:init (road l0 l54) (road l0 l78) (road l1 l25) (road l1 l85) (road l10 l25) (road l10 l43) (road l11 l51) (road l11 l72) (road l12 l30) (road l12 l48) (road l13 l38) (road l13 l48) (road l14 l40) (road l14 l86) (road l15 l37) (road l15 l78) (road l16 l30) (road l16 l92) (road l17 l61) (road l17 l71) (road l18 l20) (road l18 l94) (road l19 l55) (road l19 l71) (road l2 l29) (road l2 l31) (road l20 l18) (road l20 l84) (road l21 l32) (road l21 l69) (road l22 l38) (road l22 l51) (road l23 l3) (road l23 l69) (road l24 l48) (road l24 l8) (road l25 l1) (road l25 l10) (road l26 l35) (road l26 l80) (road l27 l31) (road l27 l45) (road l28 l47) (road l28 l59) (road l29 l2) (road l29 l43) (road l3 l23) (road l3 l92) (road l30 l12) (road l30 l16) (road l30 l42) (road l31 l2) (road l31 l27) (road l31 l75) (road l32 l21) (road l32 l36) (road l33 l89) (road l34 l37) (road l34 l41) (road l35 l26) (road l35 l52) (road l36 l32) (road l36 l58) (road l36 l62) (road l37 l15) (road l37 l34) (road l38 l13) (road l38 l22) (road l39 l45) (road l39 l50) (road l4 l44) (road l4 l5) (road l40 l14) (road l40 l82) (road l41 l34) (road l41 l89) (road l42 l30) (road l42 l53) (road l43 l10) (road l43 l29) (road l44 l4) (road l44 l56) (road l45 l27) (road l45 l39) (road l46 l68) (road l46 l93) (road l47 l28) (road l47 l6) (road l47 l65) (road l48 l12) (road l48 l13) (road l48 l24) (road l48 l81) (road l49 l50) (road l49 l64) (road l5 l4) (road l5 l91) (road l50 l39) (road l50 l49) (road l51 l11) (road l51 l22) (road l52 l35) (road l52 l54) (road l52 l63) (road l53 l42) (road l53 l70) (road l54 l0) (road l54 l52) (road l54 l57) (road l54 l74) (road l54 l78) (road l55 l19) (road l55 l77) (road l56 l44) (road l56 l72) (road l57 l54) (road l57 l66) (road l58 l36) (road l58 l90) (road l59 l28) (road l59 l76) (road l59 l87) (road l6 l47) (road l6 l95) (road l60 l75) (road l60 l90) (road l61 l17) (road l61 l9) (road l62 l36) (road l62 l88) (road l63 l52) (road l63 l81) (road l64 l49) (road l64 l77) (road l65 l47) (road l65 l66) (road l66 l57) (road l66 l65) (road l67 l79) (road l67 l93) (road l68 l46) (road l68 l9) (road l69 l21) (road l69 l23) (road l7 l76) (road l7 l96) (road l70 l53) (road l70 l74) (road l71 l17) (road l71 l19) (road l71 l91) (road l71 l95) (road l72 l11) (road l72 l56) (road l73 l8) (road l73 l94) (road l74 l54) (road l74 l70) (road l75 l31) (road l75 l60) (road l76 l59) (road l76 l7) (road l77 l55) (road l77 l64) (road l78 l0) (road l78 l15) (road l78 l54) (road l78 l83) (road l79 l67) (road l79 l86) (road l8 l24) (road l8 l73) (road l80 l26) (road l81 l48) (road l81 l63) (road l82 l40) (road l82 l87) (road l82 l96) (road l83 l78) (road l83 l84) (road l84 l20) (road l84 l83) (road l85 l1) (road l85 l88) (road l86 l14) (road l86 l79) (road l87 l59) (road l87 l82) (road l88 l62) (road l88 l85) (road l89 l33) (road l89 l41) (road l9 l61) (road l9 l68) (road l90 l58) (road l90 l60) (road l91 l5) (road l91 l71) (road l92 l16) (road l92 l3) (road l93 l46) (road l93 l67) (road l94 l18) (road l94 l73) (road l95 l6) (road l95 l71) (road l96 l7) (road l96 l82) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l22) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l31) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l46) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l69) (spare-in l72) (spare-in l75) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l83) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l92) (spare-in l93) (vehicle-at l33))
    (:goal (vehicle-at l80))
)