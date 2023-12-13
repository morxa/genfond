(define (problem tireworld-100-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 l99 - location)
    (:init (road l0 l55) (road l0 l80) (road l1 l72) (road l1 l97) (road l10 l71) (road l10 l73) (road l11 l52) (road l11 l86) (road l12 l46) (road l12 l53) (road l13 l24) (road l13 l98) (road l14 l23) (road l14 l6) (road l15 l59) (road l15 l91) (road l16 l19) (road l16 l9) (road l17 l68) (road l17 l94) (road l18 l82) (road l18 l92) (road l19 l16) (road l19 l30) (road l2 l26) (road l2 l55) (road l20 l35) (road l20 l81) (road l21 l23) (road l21 l45) (road l22 l29) (road l22 l65) (road l23 l14) (road l23 l21) (road l24 l13) (road l24 l8) (road l25 l37) (road l25 l86) (road l26 l2) (road l26 l5) (road l26 l9) (road l27 l42) (road l27 l76) (road l28 l40) (road l28 l60) (road l29 l22) (road l29 l33) (road l3 l83) (road l3 l87) (road l30 l19) (road l30 l79) (road l31 l79) (road l31 l88) (road l32 l41) (road l32 l53) (road l32 l96) (road l33 l29) (road l33 l77) (road l34 l58) (road l34 l64) (road l35 l20) (road l35 l63) (road l36 l39) (road l36 l82) (road l37 l25) (road l37 l67) (road l38 l4) (road l38 l40) (road l39 l36) (road l39 l45) (road l4 l38) (road l4 l84) (road l40 l28) (road l40 l38) (road l41 l32) (road l41 l50) (road l41 l66) (road l42 l27) (road l42 l56) (road l43 l74) (road l43 l75) (road l44 l81) (road l44 l99) (road l45 l21) (road l45 l39) (road l46 l12) (road l46 l67) (road l47 l52) (road l47 l87) (road l48 l89) (road l48 l92) (road l49 l8) (road l49 l90) (road l5 l26) (road l5 l73) (road l50 l41) (road l50 l94) (road l51 l85) (road l51 l89) (road l52 l11) (road l52 l47) (road l53 l12) (road l53 l32) (road l53 l84) (road l54 l6) (road l54 l96) (road l55 l0) (road l55 l2) (road l56 l42) (road l56 l75) (road l57 l60) (road l57 l7) (road l58 l34) (road l58 l76) (road l59 l15) (road l59 l98) (road l6 l14) (road l6 l54) (road l60 l28) (road l60 l57) (road l61 l7) (road l61 l91) (road l62 l72) (road l62 l76) (road l63 l35) (road l63 l77) (road l64 l34) (road l64 l69) (road l65 l22) (road l65 l93) (road l66 l41) (road l66 l95) (road l67 l37) (road l67 l46) (road l68 l17) (road l68 l71) (road l69 l64) (road l69 l95) (road l7 l57) (road l7 l61) (road l70 l74) (road l70 l80) (road l71 l10) (road l71 l68) (road l72 l1) (road l72 l62) (road l73 l10) (road l73 l5) (road l74 l43) (road l74 l70) (road l75 l43) (road l75 l56) (road l76 l27) (road l76 l58) (road l76 l62) (road l77 l33) (road l77 l63) (road l78 l88) (road l78 l99) (road l79 l30) (road l79 l31) (road l8 l24) (road l8 l49) (road l80 l0) (road l80 l70) (road l81 l20) (road l81 l44) (road l82 l18) (road l82 l36) (road l83 l3) (road l83 l97) (road l84 l4) (road l84 l53) (road l85 l51) (road l85 l93) (road l86 l11) (road l86 l25) (road l87 l3) (road l87 l47) (road l88 l31) (road l88 l78) (road l89 l48) (road l89 l51) (road l89 l90) (road l9 l16) (road l9 l26) (road l90 l49) (road l90 l89) (road l91 l15) (road l91 l61) (road l92 l18) (road l92 l48) (road l93 l65) (road l93 l85) (road l94 l17) (road l94 l50) (road l95 l66) (road l95 l69) (road l96 l32) (road l96 l54) (road l97 l1) (road l97 l83) (road l98 l13) (road l98 l59) (road l99 l44) (road l99 l78) (spare-in l1) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l76) (spare-in l78) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l97) (spare-in l99) (vehicle-at l64))
    (:goal (vehicle-at l37))
)