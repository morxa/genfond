(define (problem tireworld-100-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 l99 - location)
    (:init (road l0 l64) (road l0 l74) (road l1 l65) (road l1 l67) (road l10 l16) (road l10 l59) (road l11 l19) (road l11 l77) (road l12 l34) (road l12 l62) (road l13 l14) (road l13 l99) (road l14 l13) (road l14 l56) (road l15 l47) (road l15 l76) (road l16 l10) (road l16 l85) (road l17 l41) (road l17 l54) (road l18 l2) (road l18 l67) (road l19 l11) (road l19 l48) (road l2 l18) (road l2 l41) (road l20 l44) (road l20 l71) (road l21 l63) (road l21 l70) (road l22 l29) (road l22 l36) (road l23 l4) (road l23 l87) (road l24 l5) (road l24 l56) (road l25 l36) (road l25 l50) (road l26 l70) (road l26 l98) (road l27 l45) (road l27 l52) (road l28 l32) (road l28 l49) (road l29 l22) (road l29 l79) (road l3 l31) (road l3 l74) (road l30 l51) (road l30 l62) (road l31 l3) (road l31 l38) (road l32 l28) (road l32 l82) (road l33 l73) (road l33 l96) (road l34 l12) (road l34 l39) (road l35 l57) (road l35 l91) (road l36 l22) (road l36 l25) (road l37 l98) (road l37 l99) (road l38 l31) (road l38 l60) (road l39 l34) (road l39 l97) (road l4 l23) (road l4 l58) (road l40 l42) (road l40 l92) (road l41 l17) (road l41 l2) (road l42 l40) (road l42 l7) (road l43 l61) (road l43 l8) (road l44 l20) (road l44 l81) (road l45 l27) (road l45 l46) (road l46 l45) (road l46 l9) (road l47 l15) (road l47 l79) (road l48 l19) (road l48 l55) (road l49 l28) (road l49 l55) (road l5 l24) (road l5 l82) (road l50 l25) (road l50 l90) (road l51 l30) (road l51 l88) (road l52 l27) (road l52 l75) (road l53 l63) (road l53 l78) (road l54 l17) (road l54 l80) (road l55 l48) (road l55 l49) (road l56 l14) (road l56 l24) (road l57 l35) (road l57 l8) (road l58 l4) (road l58 l83) (road l59 l10) (road l59 l89) (road l6 l7) (road l6 l84) (road l60 l38) (road l60 l91) (road l61 l43) (road l61 l96) (road l62 l12) (road l62 l30) (road l63 l21) (road l63 l53) (road l64 l0) (road l64 l92) (road l65 l1) (road l65 l93) (road l66 l75) (road l66 l95) (road l67 l1) (road l67 l18) (road l68 l78) (road l68 l95) (road l69 l72) (road l69 l9) (road l7 l42) (road l7 l6) (road l70 l21) (road l70 l26) (road l71 l20) (road l71 l83) (road l72 l69) (road l72 l94) (road l73 l33) (road l73 l97) (road l74 l0) (road l74 l3) (road l75 l52) (road l75 l66) (road l76 l15) (road l77 l11) (road l78 l53) (road l78 l68) (road l79 l29) (road l79 l47) (road l8 l43) (road l8 l57) (road l80 l54) (road l80 l86) (road l81 l44) (road l81 l86) (road l82 l32) (road l82 l5) (road l83 l58) (road l83 l71) (road l84 l6) (road l84 l93) (road l85 l16) (road l85 l94) (road l86 l80) (road l86 l81) (road l87 l23) (road l87 l89) (road l88 l51) (road l88 l90) (road l89 l59) (road l89 l87) (road l9 l46) (road l9 l69) (road l90 l50) (road l90 l88) (road l91 l35) (road l91 l60) (road l92 l40) (road l92 l64) (road l93 l65) (road l93 l84) (road l94 l72) (road l94 l85) (road l95 l66) (road l95 l68) (road l96 l33) (road l96 l61) (road l97 l39) (road l97 l73) (road l98 l26) (road l98 l37) (road l99 l13) (road l99 l37) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l97) (spare-in l98) (spare-in l99) (vehicle-at l76))
    (:goal (vehicle-at l77))
)