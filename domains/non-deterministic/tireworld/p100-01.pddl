(define (problem tireworld-100-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 l99 - location)
    (:init (road l0 l53) (road l0 l89) (road l1 l39) (road l1 l83) (road l10 l12) (road l10 l94) (road l11 l39) (road l11 l6) (road l12 l10) (road l12 l91) (road l13 l34) (road l13 l9) (road l14 l18) (road l14 l97) (road l15 l35) (road l15 l46) (road l15 l85) (road l16 l7) (road l16 l77) (road l17 l19) (road l17 l73) (road l18 l14) (road l18 l55) (road l19 l17) (road l19 l48) (road l2 l52) (road l2 l57) (road l20 l25) (road l20 l79) (road l21 l61) (road l21 l66) (road l22 l30) (road l22 l78) (road l23 l92) (road l23 l98) (road l24 l36) (road l24 l76) (road l25 l20) (road l25 l30) (road l26 l5) (road l26 l51) (road l27 l55) (road l27 l82) (road l28 l90) (road l28 l95) (road l29 l56) (road l29 l86) (road l3 l43) (road l3 l67) (road l30 l22) (road l30 l25) (road l31 l81) (road l31 l90) (road l32 l60) (road l32 l91) (road l33 l44) (road l33 l46) (road l33 l49) (road l34 l13) (road l34 l4) (road l34 l58) (road l35 l15) (road l35 l4) (road l36 l24) (road l36 l88) (road l37 l8) (road l37 l93) (road l38 l68) (road l38 l80) (road l39 l1) (road l39 l11) (road l4 l34) (road l4 l35) (road l40 l49) (road l40 l70) (road l41 l74) (road l41 l76) (road l42 l50) (road l42 l8) (road l43 l3) (road l43 l72) (road l44 l33) (road l44 l6) (road l44 l69) (road l44 l88) (road l45 l65) (road l45 l68) (road l46 l15) (road l46 l33) (road l47 l74) (road l47 l87) (road l48 l19) (road l48 l51) (road l49 l33) (road l49 l40) (road l5 l26) (road l5 l59) (road l50 l42) (road l50 l80) (road l51 l26) (road l51 l48) (road l52 l2) (road l52 l65) (road l53 l0) (road l53 l83) (road l54 l75) (road l54 l99) (road l55 l18) (road l55 l27) (road l56 l29) (road l56 l73) (road l57 l2) (road l57 l84) (road l58 l34) (road l58 l59) (road l59 l5) (road l59 l58) (road l6 l11) (road l6 l44) (road l60 l32) (road l60 l64) (road l61 l21) (road l61 l93) (road l62 l64) (road l62 l96) (road l63 l9) (road l63 l96) (road l64 l60) (road l64 l62) (road l65 l45) (road l65 l52) (road l66 l21) (road l66 l67) (road l67 l3) (road l67 l66) (road l68 l38) (road l68 l45) (road l69 l44) (road l69 l85) (road l7 l16) (road l7 l81) (road l70 l40) (road l70 l78) (road l70 l87) (road l71 l84) (road l71 l92) (road l72 l43) (road l72 l78) (road l73 l17) (road l73 l56) (road l74 l41) (road l74 l47) (road l75 l54) (road l75 l77) (road l76 l24) (road l76 l41) (road l77 l16) (road l77 l75) (road l78 l22) (road l78 l70) (road l78 l72) (road l79 l20) (road l8 l37) (road l8 l42) (road l80 l38) (road l80 l50) (road l81 l31) (road l81 l7) (road l82 l27) (road l82 l98) (road l83 l1) (road l83 l53) (road l84 l57) (road l84 l71) (road l85 l15) (road l85 l69) (road l86 l29) (road l86 l95) (road l87 l47) (road l87 l70) (road l88 l36) (road l88 l44) (road l89 l0) (road l89 l94) (road l9 l13) (road l9 l63) (road l90 l28) (road l90 l31) (road l91 l12) (road l91 l32) (road l92 l23) (road l92 l71) (road l93 l37) (road l93 l61) (road l94 l10) (road l94 l89) (road l95 l28) (road l95 l86) (road l96 l62) (road l96 l63) (road l97 l14) (road l97 l99) (road l98 l23) (road l98 l82) (road l99 l54) (road l99 l97) (spare-in l0) (spare-in l10) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l25) (spare-in l26) (spare-in l3) (spare-in l30) (spare-in l34) (spare-in l35) (spare-in l38) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l53) (spare-in l6) (spare-in l61) (spare-in l63) (spare-in l65) (spare-in l67) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l76) (spare-in l78) (spare-in l8) (spare-in l80) (spare-in l87) (spare-in l90) (spare-in l93) (spare-in l98) (spare-in l99) (vehicle-at l41))
    (:goal (vehicle-at l79))
)