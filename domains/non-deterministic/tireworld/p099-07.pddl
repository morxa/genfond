(define (problem tireworld-099-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 - location)
    (:init (road l0 l27) (road l0 l63) (road l1 l33) (road l1 l59) (road l10 l38) (road l10 l74) (road l11 l31) (road l11 l67) (road l12 l30) (road l12 l71) (road l13 l64) (road l13 l92) (road l14 l62) (road l14 l91) (road l15 l21) (road l15 l38) (road l16 l3) (road l16 l7) (road l17 l22) (road l17 l45) (road l18 l56) (road l18 l68) (road l19 l82) (road l19 l92) (road l2 l27) (road l2 l36) (road l20 l31) (road l20 l53) (road l21 l15) (road l21 l23) (road l22 l17) (road l22 l72) (road l23 l21) (road l23 l4) (road l24 l51) (road l24 l55) (road l25 l41) (road l25 l5) (road l26 l66) (road l26 l75) (road l27 l0) (road l27 l2) (road l28 l34) (road l28 l42) (road l29 l44) (road l29 l71) (road l3 l16) (road l3 l73) (road l30 l12) (road l30 l76) (road l31 l11) (road l31 l20) (road l32 l50) (road l32 l85) (road l33 l1) (road l33 l45) (road l34 l28) (road l34 l52) (road l35 l42) (road l35 l47) (road l36 l2) (road l36 l8) (road l37 l79) (road l37 l96) (road l38 l10) (road l38 l15) (road l39 l69) (road l39 l74) (road l4 l23) (road l4 l49) (road l40 l58) (road l40 l86) (road l41 l25) (road l41 l58) (road l42 l28) (road l42 l35) (road l43 l54) (road l43 l73) (road l44 l29) (road l44 l77) (road l45 l17) (road l45 l33) (road l45 l64) (road l46 l57) (road l46 l70) (road l47 l35) (road l47 l5) (road l48 l69) (road l48 l89) (road l49 l4) (road l49 l80) (road l5 l25) (road l5 l47) (road l50 l32) (road l50 l83) (road l51 l24) (road l51 l60) (road l52 l34) (road l52 l9) (road l53 l20) (road l53 l89) (road l54 l43) (road l54 l59) (road l55 l24) (road l55 l56) (road l56 l18) (road l56 l55) (road l57 l46) (road l57 l71) (road l58 l40) (road l58 l41) (road l59 l1) (road l59 l54) (road l6 l81) (road l6 l97) (road l60 l51) (road l60 l81) (road l61 l75) (road l61 l78) (road l62 l14) (road l62 l67) (road l63 l0) (road l63 l68) (road l64 l13) (road l64 l45) (road l65 l77) (road l65 l87) (road l66 l26) (road l66 l90) (road l67 l11) (road l67 l62) (road l68 l18) (road l68 l63) (road l69 l39) (road l69 l48) (road l7 l16) (road l7 l88) (road l70 l46) (road l70 l72) (road l71 l12) (road l71 l29) (road l71 l57) (road l72 l22) (road l72 l70) (road l73 l3) (road l73 l43) (road l74 l10) (road l74 l39) (road l75 l26) (road l75 l61) (road l76 l30) (road l76 l84) (road l77 l44) (road l77 l65) (road l78 l61) (road l78 l94) (road l79 l37) (road l79 l98) (road l8 l36) (road l8 l88) (road l80 l49) (road l80 l98) (road l81 l6) (road l81 l60) (road l82 l19) (road l83 l50) (road l83 l94) (road l84 l76) (road l85 l32) (road l85 l97) (road l86 l40) (road l86 l93) (road l87 l65) (road l87 l9) (road l88 l7) (road l88 l8) (road l89 l48) (road l89 l53) (road l9 l52) (road l9 l87) (road l90 l66) (road l90 l95) (road l91 l14) (road l91 l95) (road l92 l13) (road l92 l19) (road l93 l86) (road l93 l96) (road l94 l78) (road l94 l83) (road l95 l90) (road l95 l91) (road l96 l37) (road l96 l93) (road l97 l6) (road l97 l85) (road l98 l79) (road l98 l80) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l56) (spare-in l57) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l81) (spare-in l83) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l97) (spare-in l98) (vehicle-at l82))
    (:goal (vehicle-at l84))
)