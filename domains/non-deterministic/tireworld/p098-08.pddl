(define (problem tireworld-098-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 - location)
    (:init (road l0 l17) (road l0 l48) (road l1 l23) (road l1 l61) (road l10 l29) (road l10 l65) (road l11 l22) (road l11 l61) (road l12 l50) (road l12 l94) (road l13 l28) (road l13 l39) (road l14 l43) (road l14 l95) (road l15 l29) (road l15 l84) (road l16 l21) (road l16 l89) (road l17 l0) (road l17 l54) (road l18 l21) (road l18 l80) (road l19 l31) (road l19 l35) (road l2 l71) (road l2 l9) (road l20 l27) (road l20 l33) (road l20 l68) (road l21 l16) (road l21 l18) (road l21 l64) (road l22 l11) (road l22 l88) (road l23 l1) (road l23 l40) (road l24 l5) (road l24 l55) (road l25 l39) (road l25 l95) (road l26 l53) (road l26 l85) (road l27 l20) (road l27 l34) (road l28 l13) (road l28 l90) (road l29 l10) (road l29 l15) (road l3 l60) (road l3 l8) (road l30 l32) (road l30 l36) (road l31 l19) (road l31 l76) (road l32 l30) (road l32 l62) (road l33 l20) (road l33 l77) (road l34 l27) (road l34 l45) (road l35 l19) (road l35 l41) (road l36 l30) (road l36 l45) (road l37 l63) (road l37 l67) (road l38 l55) (road l38 l66) (road l38 l79) (road l38 l91) (road l39 l13) (road l39 l25) (road l4 l43) (road l4 l85) (road l40 l23) (road l40 l83) (road l41 l35) (road l41 l52) (road l42 l80) (road l42 l93) (road l43 l14) (road l43 l4) (road l44 l62) (road l44 l70) (road l45 l34) (road l45 l36) (road l45 l71) (road l46 l79) (road l46 l82) (road l47 l66) (road l47 l74) (road l48 l0) (road l48 l96) (road l49 l63) (road l49 l80) (road l5 l24) (road l5 l77) (road l50 l12) (road l50 l6) (road l51 l86) (road l51 l87) (road l52 l41) (road l52 l94) (road l53 l26) (road l53 l54) (road l54 l17) (road l54 l53) (road l55 l24) (road l55 l38) (road l56 l64) (road l56 l70) (road l57 l69) (road l57 l81) (road l58 l73) (road l58 l81) (road l59 l67) (road l59 l73) (road l6 l50) (road l6 l92) (road l60 l3) (road l61 l1) (road l61 l11) (road l62 l32) (road l62 l44) (road l63 l37) (road l63 l49) (road l64 l21) (road l64 l56) (road l64 l8) (road l65 l10) (road l65 l78) (road l65 l97) (road l66 l38) (road l66 l47) (road l67 l37) (road l67 l59) (road l68 l20) (road l68 l7) (road l69 l57) (road l69 l88) (road l7 l68) (road l7 l75) (road l70 l44) (road l70 l56) (road l71 l2) (road l71 l45) (road l72 l84) (road l72 l9) (road l73 l58) (road l73 l59) (road l74 l47) (road l74 l90) (road l75 l7) (road l75 l96) (road l76 l31) (road l76 l87) (road l77 l33) (road l77 l5) (road l78 l65) (road l78 l80) (road l79 l38) (road l79 l46) (road l8 l3) (road l8 l64) (road l80 l18) (road l80 l42) (road l80 l49) (road l80 l78) (road l81 l57) (road l81 l58) (road l82 l46) (road l82 l92) (road l83 l40) (road l83 l86) (road l84 l15) (road l84 l72) (road l85 l26) (road l85 l4) (road l86 l51) (road l86 l83) (road l87 l51) (road l87 l76) (road l88 l22) (road l88 l69) (road l89 l16) (road l89 l92) (road l9 l2) (road l9 l72) (road l90 l28) (road l90 l74) (road l91 l38) (road l92 l6) (road l92 l82) (road l92 l89) (road l93 l42) (road l93 l97) (road l94 l12) (road l94 l52) (road l95 l14) (road l95 l25) (road l96 l48) (road l96 l75) (road l97 l65) (road l97 l93) (spare-in l1) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l51) (spare-in l54) (spare-in l56) (spare-in l58) (spare-in l6) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l70) (spare-in l72) (spare-in l75) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l82) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l90) (spare-in l92) (spare-in l93) (spare-in l95) (vehicle-at l60))
    (:goal (vehicle-at l91))
)