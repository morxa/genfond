(define (problem tireworld-099-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 - location)
    (:init (road l0 l55) (road l0 l6) (road l1 l56) (road l1 l64) (road l10 l34) (road l10 l56) (road l10 l77) (road l11 l23) (road l12 l40) (road l12 l76) (road l13 l49) (road l13 l57) (road l14 l71) (road l14 l72) (road l15 l21) (road l15 l52) (road l16 l26) (road l16 l40) (road l17 l75) (road l17 l97) (road l18 l67) (road l18 l82) (road l19 l27) (road l2 l58) (road l2 l66) (road l20 l30) (road l20 l64) (road l21 l15) (road l21 l98) (road l22 l30) (road l22 l93) (road l23 l11) (road l23 l49) (road l24 l60) (road l24 l70) (road l25 l35) (road l25 l42) (road l26 l16) (road l26 l53) (road l27 l19) (road l27 l92) (road l28 l95) (road l28 l98) (road l29 l77) (road l29 l97) (road l3 l44) (road l3 l93) (road l30 l20) (road l30 l22) (road l31 l35) (road l31 l80) (road l32 l67) (road l32 l88) (road l33 l38) (road l33 l5) (road l34 l10) (road l34 l69) (road l35 l25) (road l35 l31) (road l36 l4) (road l36 l62) (road l37 l81) (road l37 l96) (road l38 l33) (road l38 l7) (road l39 l58) (road l39 l63) (road l4 l36) (road l4 l48) (road l4 l83) (road l40 l12) (road l40 l16) (road l41 l83) (road l41 l87) (road l42 l25) (road l42 l8) (road l43 l87) (road l43 l91) (road l44 l3) (road l44 l62) (road l45 l91) (road l45 l95) (road l46 l59) (road l46 l90) (road l47 l82) (road l47 l89) (road l48 l4) (road l48 l7) (road l49 l13) (road l49 l23) (road l5 l33) (road l5 l64) (road l50 l61) (road l50 l63) (road l51 l53) (road l51 l84) (road l52 l15) (road l52 l89) (road l53 l26) (road l53 l51) (road l54 l72) (road l54 l74) (road l55 l0) (road l55 l78) (road l56 l1) (road l56 l10) (road l57 l13) (road l57 l68) (road l58 l2) (road l58 l39) (road l59 l46) (road l59 l68) (road l6 l0) (road l6 l9) (road l60 l24) (road l60 l65) (road l61 l50) (road l61 l85) (road l62 l36) (road l62 l44) (road l63 l39) (road l63 l50) (road l64 l1) (road l64 l20) (road l64 l5) (road l65 l60) (road l65 l89) (road l66 l2) (road l66 l9) (road l67 l18) (road l67 l32) (road l68 l57) (road l68 l59) (road l69 l34) (road l69 l70) (road l7 l38) (road l7 l48) (road l70 l24) (road l70 l69) (road l71 l14) (road l71 l84) (road l72 l14) (road l72 l54) (road l73 l80) (road l73 l96) (road l74 l54) (road l74 l88) (road l75 l17) (road l75 l93) (road l76 l12) (road l76 l79) (road l77 l10) (road l77 l29) (road l78 l55) (road l78 l94) (road l79 l76) (road l79 l86) (road l8 l42) (road l8 l91) (road l80 l31) (road l80 l73) (road l81 l37) (road l81 l90) (road l82 l18) (road l82 l47) (road l83 l4) (road l83 l41) (road l84 l51) (road l84 l71) (road l85 l61) (road l85 l92) (road l86 l79) (road l86 l94) (road l86 l97) (road l87 l41) (road l87 l43) (road l88 l32) (road l88 l74) (road l89 l47) (road l89 l52) (road l89 l65) (road l9 l6) (road l9 l66) (road l90 l46) (road l90 l81) (road l91 l43) (road l91 l45) (road l91 l8) (road l92 l27) (road l92 l85) (road l93 l22) (road l93 l3) (road l93 l75) (road l94 l78) (road l94 l86) (road l95 l28) (road l95 l45) (road l96 l37) (road l96 l73) (road l97 l17) (road l97 l29) (road l97 l86) (road l98 l21) (road l98 l28) (spare-in l0) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l2) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l31) (spare-in l32) (spare-in l35) (spare-in l37) (spare-in l39) (spare-in l40) (spare-in l42) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l61) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l98) (vehicle-at l19))
    (:goal (vehicle-at l11))
)